import torch
from .transformer import Transformer
from .config import config
from .tensors import LearnedSinusoidalPosEmb
from xformers.ops import fmha
from torchdiffeq import odeint

class SupervoceVariant1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Parameters
        self.n_dim = 1024
        self.max_seq_len = 8 * 1024
        
        # Positional embeddings
        self.positional_embedding_text = torch.nn.Embedding(self.max_seq_len, self.n_dim)
        torch.nn.init.normal_(self.positional_embedding_text.weight, mean=0.0, std=0.02)
        self.positional_embedding_audio = torch.nn.Embedding(self.max_seq_len, self.n_dim)
        torch.nn.init.normal_(self.positional_embedding_audio.weight, mean=0.0, std=0.02)

        # Text Condition
        self.text_embedding = torch.nn.Embedding(8 * 1024, self.n_dim)
        torch.nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)

        # Audio embedding
        self.audio_embedding = torch.nn.Embedding(1024 + 1, self.n_dim)

        # Input projection
        self.input_projection = torch.nn.Linear(self.n_dim * 2, self.n_dim, bias=False)

        # Transformer
        self.transformer = Transformer(
            n_heads = 16,
            n_layers = 12,
            n_dim = self.n_dim,
            n_dim_head = 16, # n_dim // n_heads
            n_dim_ffn = 4096,
            att_dropout = 0,
            ffn_dropout = 0.1
        )
        
        # Prediction
        self.prediction = torch.nn.Linear(self.n_dim, 1024)
        torch.nn.init.normal_(self.prediction.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.prediction.bias)


    def forward(self, *, condition_text, condition_audio, duration, target = None):
        device = condition_text[0].device
        
        #
        # Check shapes
        #

        B = len(condition_text)
        assert len(condition_audio) == B
        assert len(duration) == B
        if target is not None:
            assert len(target) == B
            assert all(t.shape[0] + c.shape[0] == d for t, d, c in zip(target, duration, condition_audio))
        for i in range(B):
            assert len(condition_text[i].shape) == 1, condition_text[i].shape
            assert len(condition_audio[i].shape) == 1, condition_audio[i].shape
            assert condition_text[i].shape[0] <= duration[i]
            assert condition_audio[i].shape[0] <= duration[i]
            if target is not None:
                assert target[i].shape[0] + condition_audio[i].shape[0] == duration[i]

        #
        # Prepare inputs
        #

        # Pad inputs
        inputs_text = []
        inputs_audio = []
        inputs_positional = []
        for i in range(B):
            d = duration[i]
            inputs_text.append(torch.nn.functional.pad(condition_text[i], (0, d - condition_text[i].shape[0]), "constant", 0))
            inputs_audio.append(torch.nn.functional.pad(condition_audio[i] + 1, (0, d - condition_audio[i].shape[0]), "constant", 0))
            inputs_positional.append(torch.arange(d).to(device, non_blocking=True))

        # Cat everything
        inputs_text = torch.cat(inputs_text)
        inputs_audio = torch.cat(inputs_audio)

        # Embeddings
        inputs_text = self.text_embedding(inputs_text)
        inputs_audio = self.audio_embedding(inputs_audio)

        # Positional embeddings
        inputs_positional = torch.cat(inputs_positional)
        inputs_text += self.positional_embedding_text(inputs_positional)
        inputs_audio += self.positional_embedding_audio(inputs_positional)

        # Input projection
        inputs = torch.cat([inputs_text, inputs_audio], dim=-1)
        inputs = self.input_projection(inputs)

        #
        # Run transformer
        #
        mask = fmha.BlockDiagonalMask.from_seqlens(duration)
        x = self.transformer(inputs.unsqueeze(0), mask = mask).squeeze(0)

        #
        # Predict
        #

        x = self.prediction(x)

        #
        # Split predictions
        #

        predicted = []
        offset = 0
        for i in range(B):
            predicted.append(x[offset + (duration[i] - target[i].shape[0]): offset + duration[i]])
            offset += duration[i]

        #
        # Loss
        #

        if target is not None:
            loss = torch.nn.functional.cross_entropy(torch.cat(predicted), torch.cat(target))
            return predicted, loss
        else:
            return predicted


class SupervoiceVariant2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Parameters
        self.n_dim = 1024
        self.n_vocab = 8 * 1024
        self.max_seq_len = 8 * 1024

        # Positional embeddings
        self.positional_embedding_text = torch.nn.Embedding(self.max_seq_len, self.n_dim)
        torch.nn.init.normal_(self.positional_embedding_text.weight, mean=0.0, std=0.02)
        self.positional_embedding_audio = torch.nn.Embedding(self.max_seq_len, self.n_dim)
        torch.nn.init.normal_(self.positional_embedding_audio.weight, mean=0.0, std=0.02)

        # Text Embedding
        self.text_embedding = torch.nn.Embedding(self.n_vocab, self.n_dim)
        torch.nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)

        # Audio embedding
        self.audio_embedding = torch.nn.Linear(config.audio.n_mels, self.n_dim, bias=False)
        torch.nn.init.normal_(self.audio_embedding.weight, mean=0.0, std=0.02)

        # Noise embedding
        self.noise_embedding = torch.nn.Linear(config.audio.n_mels, self.n_dim, bias=False)
        torch.nn.init.normal_(self.noise_embedding.weight, mean=0.0, std=0.02)

        # Transformer input
        self.input_projection = torch.nn.Linear(3 * self.n_dim, self.n_dim, bias=False)

        # Sinusoidal positional embedding for time
        self.time_embedding = LearnedSinusoidalPosEmb(self.n_dim)

        # Transformer
        self.transformer = Transformer(
            n_heads = 16,
            n_layers = 12,
            n_dim = self.n_dim,
            n_dim_head = 16, # n_dim // n_heads
            n_dim_ffn = self.n_dim * 4,
            att_dropout = 0,
            ffn_dropout = 0.1,
            enable_skip_connections = True
        )

        # Prediction
        self.prediction = torch.nn.Linear(self.n_dim, config.audio.n_mels)

    def sample(self, *, tokens, audio, interval, steps, alpha = None):
        
        #
        # Prepare
        #

        # Mask out audio
        masked_audio = audio.clone()
        masked_audio[interval[0]: interval[1]] = 0

        # Create noise
        noise = torch.randn_like(audio)

        # Create time interpolation
        times = torch.linspace(0, 1, steps, device = audio.device)

        #
        # Solver
        #

        def solver(t, z):

            # If alpha is not provided
            if alpha is None:
                return self.forward(
                    condition_text = [tokens], 
                    condition_audio = [masked_audio], 
                    noisy_audio = [z], 
                    times = [t],
                )[0]

            # If alpha is provided - zero out tokens and audio and mix together
            tokens_empty = torch.zeros_like(tokens)
            audio_empty = torch.zeros_like(audio)

            # Inference
            predicted_mix = self.forward(
                condition_text = [torch.zeros_like(tokens), tokens], 
                condition_audio = [torch.zeros_like(audio), masked_audio], 
                noisy_audio = [z, z], 
                times = [t, t]
            )
            predicted_conditioned = predicted_mix[1]
            predicted_unconditioned = predicted_mix[0]
            
            # CFG prediction

            # There are different ways to do CFG, this is my very naive version, which worked for me:
            # prediction = (1 + alpha) * predicted_conditioned - alpha * predicted_unconditioned

            # Original paper uses a different one, but i found that it simply creates overexposed values
            # prediction = predicted_unconditioned + (predicted_conditioned - predicted_unconditioned) * alpha

            # This is from the latest paper that rescales original formula (https://arxiv.org/abs/2305.08891):
            prediction = predicted_conditioned + (predicted_conditioned - predicted_unconditioned) * alpha
            prediction_rescaled = predicted_conditioned.std() * (prediction / prediction.std())

            return prediction


        trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        #
        # Output predicted interval
        #

        return trajectory[-1][interval[0]: interval[1]]

    def forward(self, *, condition_text, condition_audio, noisy_audio, times, intervals = None, target = None):
        device = condition_text[0].device

        # Check shapes
        B = len(condition_text)
        assert len(condition_audio) == B
        assert len(noisy_audio) == B
        assert len(times) == B
        if target is not None:
            assert intervals is not None
            assert len(target) == B

        # Calculate durations
        durations = [c.shape[0] for c in condition_audio]

        # Check inner shapes
        for i in range(B):
            assert len(condition_text[i].shape) == 1, condition_text[i].shape
            assert len(condition_audio[i].shape) == 2, condition_audio[i].shape
            assert len(noisy_audio[i].shape) == 2, condition_audio[i].shape
            assert condition_text[i].shape[0] <= durations[i], condition_text[i].shape[0]
            assert condition_audio[i].shape[0] == durations[i]
            assert condition_audio[i].shape[1] == config.audio.n_mels
            assert noisy_audio[i].shape[0] == durations[i]
            assert noisy_audio[i].shape[1] == config.audio.n_mels
            if target is not None:
                assert len(intervals[i]) == 2
                assert intervals[i][0] >= 0
                assert intervals[i][1] <= durations[i]
                assert intervals[i][0] <= intervals[i][1]
                assert target[i].shape[0] == intervals[i][1] - intervals[i][0]
                assert target[i].shape[1] == config.audio.n_mels

        # Prepare inputs
        inputs_text = []
        inputs_positional = []
        for i in range(B):
            d = durations[i]
            inputs_text.append(torch.nn.functional.pad(condition_text[i], (0, d - condition_text[i].shape[0]), "constant", 0))
            inputs_positional.append(torch.arange(d).to(device, non_blocking=True))

        # Cat everything
        inputs_positional = torch.cat(inputs_positional)
        inputs_text = torch.cat(inputs_text)
        inputs_audio = torch.cat(condition_audio)
        inputs_noisy = torch.cat(noisy_audio)

        # Text
        inputs_text = self.text_embedding(inputs_text)
        inputs_text += self.positional_embedding_text(inputs_positional)

        # Audio
        inputs_audio = self.audio_embedding(inputs_audio)
        inputs_audio += self.positional_embedding_audio(inputs_positional)
        inputs_noisy = self.noise_embedding(inputs_noisy)
        inputs_noisy += self.positional_embedding_audio(inputs_positional)

        # Input projection
        inputs = torch.cat([inputs_text, inputs_audio, inputs_noisy], dim=-1)
        inputs = self.input_projection(inputs)

        # Time embeddings
        times = self.time_embedding(times)

        # Merge time embeddings
        inputs_timed = []
        offset = 0
        for i in range(B):
            d = durations[i]
            inputs_timed.append(torch.cat([inputs[offset: offset + d], times[i].unsqueeze(0)], dim=0))
            offset += d
        inputs = torch.cat(inputs_timed)

        # Transformer
        attention_mask = None
        if len(durations) > 1: # Disable mask for speed if not batched
            attention_mask = fmha.BlockDiagonalMask.from_seqlens([i + 1 for i in durations])
        x = self.transformer(inputs.unsqueeze(0), mask = attention_mask).squeeze(0)
        
        # Predict
        x = self.prediction(x)

        # Split predictions
        outputs = []
        offset = 0
        for i in range(B):
            outputs.append(x[offset: offset + durations[i]])
            offset += durations[i] + 1 # +1 for time embedding

        # Compute loss
        if target is not None:

            # Compute target intervals
            preds = []
            offset = 0
            for i in range(B):
                preds.append(x[offset + intervals[i][0]: offset + intervals[i][1]])
                offset += durations[i] + 1 # +1 for time embedding

            # Compute loss
            target = torch.cat(target, dim = 0)
            predd_cat = torch.cat(preds, dim = 0)
            loss = torch.nn.functional.mse_loss(predd_cat, target)

            # Normalize by number of frames
            loss = loss / target.shape[0]

            return outputs, loss
        
        return outputs
