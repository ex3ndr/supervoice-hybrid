# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
from tqdm import tqdm
import time
from contextlib import nullcontext
from pathlib import Path
import shutil
import math
import random

# ML
import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.profiler import profile, record_function, ProfilerActivity

# Local
from supervoice_hybrid import SupervoceHybridStage1, SentencePieceTextTokenizer
from train.dataset import load_encodec_sampler, create_async_loader

# Experiment
train_experiment = "exp-2"
train_project="supervoice-e2"
train_auto_resume = True

# Training schedule and parameters
train_target_batch_size = 16
train_batch_size = 12
train_mixed_precision = "fp16" # "bf16" or "fp16" or None
train_clip_grad_norm = 1 # Common reproductions are using 100 or 1
train_lr_start = 1e-12
train_lr_max = 5e-4
train_steps = 600000
train_warmup_steps = 32000 # I am using faster warmup - it is more natural for me after working on voicebox

# Utilities
train_loader_workers = 32
train_log_every = 1
train_save_every = 1000
train_watch_every = 1000

#
# Factory
#

def create_sampler():
    # tokenizer = UnitTextTokenizer()
    tokenizer = SentencePieceTextTokenizer("./tokenizer_text.model")
    # train_sampler = load_encodec_sampler("./external_datasets/libriheavy/libriheavy_cuts_small.jsonl.gz", "./external_datasets/libriheavy-encodec/", train_batch_size, tokenizer)
    # train_sampler = load_encodec_sampler("./external_datasets/libriheavy/libriheavy_cuts_medium.jsonl.gz", "./external_datasets/libriheavy-medium-encodec/", train_batch_size, tokenizer)
    train_sampler = load_encodec_sampler("./external_datasets/libriheavy/libriheavy_cuts_large.jsonl.gz", "./external_datasets/libriheavy-large-encodec/", train_batch_size, tokenizer)
    return train_sampler

def create_model():
    return SupervoceHybridStage1()

def do_train(accelerator, model, inputs):
    device = accelerator.device
    audio, text = inputs
    
    # Reshape inputs
    condition_text = []
    condition_audio = []
    targets = []
    durations = []
    for B in range(len(audio)):
        a = audio[B].squeeze(0)[0]
        t = text[B].squeeze(0)
        
        # Calculate split
        min_duration = 0
        max_duration = a.shape[0] // 3
        audio_split = random.randint(min_duration, max_duration)

        # Append
        condition_text.append(t.to(device, non_blocking=True))
        condition_audio.append(a[:audio_split].to(device, non_blocking=True))
        targets.append(a[audio_split:].to(device, non_blocking=True))
        durations.append(a.shape[0])

    # Forward
    _, loss = model(
        condition_text = condition_text,
        condition_audio = condition_audio,
        duration = durations,
        target = targets
    )

    return loss

#
# Train
#

def main():

    # Calculate gradient accumulation
    train_grad_accum_every = train_target_batch_size
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        train_grad_accum_every = math.ceil(train_target_batch_size / torch.cuda.device_count())
    print(f"Running with gradient accumulation every {train_grad_accum_every}")

    # Prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every, mixed_precision=train_mixed_precision)
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if train_mixed_precision == "fp16" else (torch.bfloat16 if train_mixed_precision == "bf16" else torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    lr_start = train_lr_start * accelerator.num_processes
    lr_max = train_lr_max * accelerator.num_processes
    random_suffix = ''.join(random.choices('0123456789abcdef', k=6))
    run_id = f"{train_experiment}-{random_suffix}"

    # Prepare dataset
    accelerator.print("Loading sampler...")    
    train_sampler = create_sampler()
    train_loader = create_async_loader(train_sampler, num_workers = train_loader_workers)
    train_cycle = cycle(train_loader)

    # Model
    accelerator.print("Loading model...")
    step = 1
    model = create_model()
    raw_model = model
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], train_lr_start, betas=[0.9, 0.95],weight_decay=0.01, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)

    # Checkpoint
    checkpoint = None
    if train_auto_resume and (output_dir / f"{train_experiment}.pt").exists():
        checkpoint = torch.load(str(output_dir / f"{train_experiment}.pt"), map_location="cpu")
        step = checkpoint['step']
        run_id = checkpoint['run_id']
    
    # Accelerate
    model, optim = accelerator.prepare(model, optim)
    hps = {
        "train_lr_start": train_lr_start, 
        "train_lr_max": train_lr_max, 
        "grad_accum_every": train_grad_accum_every,
        "steps": train_steps, 
        "warmup_steps": train_warmup_steps,
        "mixed_precision": train_mixed_precision,
        "clip_grad_norm": train_clip_grad_norm,
    }
    accelerator.init_trackers(train_project, config=hps, init_kwargs={"wandb":{"name":run_id,  "id": run_id, "resume": "allow"}})
    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=train_watch_every * train_grad_accum_every)

    # Save
    def save():
        # Save step checkpoint
        fname = str(output_dir / f"{train_experiment}.pt")
        fname_step = str(output_dir / f"{train_experiment}.{step}.pt")
        torch.save({

            # Model
            'model': raw_model.state_dict(), 

            # Optimizer
            'optimizer': optim.state_dict(), 
            'scheduler': scheduler.state_dict(),
            'scaler': accelerator.scaler.state_dict(),
            'step': step,
            'run_id': run_id,

        },  fname_step)

        # Overwrite main checkpoint
        shutil.copyfile(fname_step, fname)

    # Load
    if checkpoint is not None:
        raw_model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        accelerator.scaler.load_state_dict(checkpoint['scaler'])
        accelerator. print(f'Loaded at #{step}')

    # Train step
    def train_step():
        model.train()

        # Update LR
        if step < train_warmup_steps:
            lr = (lr_start + ((lr_max - lr_start) * step) / train_warmup_steps)
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            lr = lr / accelerator.num_processes
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0] / accelerator.num_processes

        # Load batch
        for _ in range(train_grad_accum_every):
            with accelerator.accumulate(model):
                
                # Load batch
                inputs = next(train_cycle)

                # Do train
                with record_function("forward"):
                    with accelerator.autocast():
                        loss = do_train(accelerator, model, inputs)
                        loss = loss / train_grad_accum_every # Rescale loss
                        
                # Backprop
                with record_function("backward"):
                    optim.zero_grad()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_clip_grad_norm)
                    optim.step()

                    # Log skipping step
                    if optim.step_was_skipped:
                        accelerator.print("Step was skipped")
                        if torch.isnan(loss):
                            raise ValueError("Loss is NaN")
        
        return loss * train_grad_accum_every, lr

    #
    # Start Training
    #

    accelerator.print("Training started at step", step)
    while step < train_steps:
        
        # Step
        start = time.time()
        loss, lr = train_step()
        end = time.time()

        # Advance
        step = step + 1

        # Summary
        if step % train_log_every == 0:
            accelerator.log({
                "learning_rate": lr,
                "loss": loss,
                "scale": accelerator.scaler.get_scale() if accelerator.scaler is not None else 1.0
            }, step=step)
            accelerator.print(f'Step {step} | Loss: {loss} | LR: {lr} | Time: {end - start}')

        # Save
        if step % train_save_every == 0:
            save()

    # End training
    if accelerator.is_main_process:
        accelerator.print("Finishing training...")
        save()
    accelerator.end_training()
    accelerator.print('✨ Training complete!')

#
# Utility
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    

if __name__ == "__main__":
    main()
