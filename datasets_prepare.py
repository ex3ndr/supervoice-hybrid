# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
import multiprocessing
import glob
import torch
import torchaudio
import csv
import gzip
import json
import math
from pathlib import Path
from tqdm import tqdm
from supervoice_hybrid.audio import load_mono_audio, spectogram
from supervoice_hybrid.config import config

#
# Parameters
#

PARAM_WORKERS = max(torch.cuda.device_count() * 2, 4)

#
# Execution
#


def clean_text(s: str) -> str:
    table = str.maketrans("’‘，。；？！（）：-《》、“”【】", "'',.;?!(): <>/\"\"[]")
    s = s.translate(table)
    return s.strip()

def execute_parallel(args):
    process_id = multiprocessing.current_process()._identity[0]
    files, output_dir, index = args
    file = files[index]['path']
    cuts = files[index]['cuts']
    device = "cuda:" + str(process_id % torch.cuda.device_count())

    # Load audio
    source = load_mono_audio(file, config.audio.sample_rate, device=device)

    # Process cuts
    for cut in cuts:
        id, start, duration, text = cut
        wav = source[int(start * config.audio.sample_rate):int((start + duration) * config.audio.sample_rate)]

        # Encode
        spec = spectogram(wav, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.mel_norm, config.audio.mel_scale, config.audio.sample_rate)

        # Save codecs
        output_file = Path(output_dir) / Path(id + ".pt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():        
            print("File exists", output_file)
        torch.save(spec.cpu(), output_file)
        
        # Save text
        output_file = Path(output_dir) / Path(id + ".txt")
        if output_file.exists():        
            print("File exists", output_file)
        with open(output_file, "w") as f:
            f.write(text)

def execute_run():
    torch.multiprocessing.set_start_method('spawn')
    
    # Collections
    collections = (
        # ("small", "./external_datasets/libriheavy/libriheavy_cuts_small.jsonl.gz", "./external_datasets/librilight/", "./processed_datasets/librilight/"),
        # ("medium", "./external_datasets/libriheavy/libriheavy_cuts_medium.jsonl.gz", "./external_datasets/librilight-medium/", "./processed_datasets/librilight-medium/"),
        ("large", "./external_datasets/libriheavy/libriheavy_cuts_large.jsonl.gz", "./external_datasets/librilight-large/", "./processed_datasets/librilight-large/"),
    )

    for collection in collections:
        name, index_path, files_path, output_path = collection

        # Load index
        print("Loading index for collection: " + name)
        files = []
        files_map = {}
        with gzip.open(index_path, "r") as f:
            for line in f:
                cut = json.loads(line)
                start = math.floor(1000 * cut["start"]) / 1000
                duration = math.floor(1000 * cut["duration"]) / 1000

                # Load audio
                wav_id = cut["recording"]["id"]
                id = cut["supervisions"][0]["id"]
                if wav_id.startswith("small/"):
                    wav_id = wav_id[len("small/"):]
                if wav_id.startswith("medium/"):
                    wav_id = wav_id[len("medium/"):]
                if wav_id.startswith("large/"):
                    wav_id = wav_id[len("large/"):]
                if id.startswith("small/"):
                    id = id[len("small/"):]
                if id.startswith("medium/"):
                    id = id[len("medium/"):]
                if id.startswith("large/"):
                    id = id[len("large/"):]

                # Check if exists
                if (Path(output_path) / Path(id + ".pt")).exists():
                    continue

                # Load text
                text = cut["supervisions"][0]["custom"]["texts"][0]
                text = clean_text(text)
        
                # Find index
                if wav_id not in files_map:
                    files_map[wav_id] = len(files)
                    files.append({ "path": files_path + wav_id + ".flac", "cuts": []})
                index = files_map[wav_id]

                # Append
                files[index]['cuts'].append((id, start, duration, text))

        # Process files
        print("Processing files for collection: " + name)
        with multiprocessing.Manager() as manager:
            files = manager.list(files)
            args_list = [(files, output_path, i) for i in range(len(files))]
            with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
                for result in tqdm(pool.imap_unordered(execute_parallel, args_list, chunksize=32), total=len(files)):
                    pass

    # End
    print("Done")

if __name__ == "__main__":
    execute_run()