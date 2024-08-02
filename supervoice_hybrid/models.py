import torch
from .transformer import Transformer
from .config import config
from .tensors import LearnedSinusoidalPosEmb
from xformers.ops import fmha

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
            adaptive = True,
            enable_skip_connections = True
        )

        # Prediction
        self.prediction = torch.nn.Linear(self.n_dim, config.audio.n_mels)

    def forward(self, *, condition_text, condition_audio, noisy_audio, intervals, times, target = None):
        device = condition_text[0].device

        # Check shapes
        B = len(condition_text)
        assert len(condition_audio) == B
        assert len(noisy_audio) == B
        assert len(intervals) == B
        assert len(times) == B
        if target is not None:
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
            assert len(intervals[i]) == 2
            assert intervals[i][0] >= 0
            assert intervals[i][1] <= durations[i]
            assert intervals[i][0] <= intervals[i][1]
            if target is not None:
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

        # Transformer
        attention_mask = None
        if len(durations) > 1: # Disable mask for speed if not batched
            attention_mask = fmha.BlockDiagonalMask.from_seqlens(durations)
        x = self.transformer(inputs.unsqueeze(0), mask = attention_mask, condition = times.unsqueeze(0)).squeeze(0)
        
        # Predict
        x = self.prediction(x)

        # Split predictions
        preds = []
        offset = 0
        for i in range(B):
            preds.append(x[offset + intervals[0]: offset + intervals[1]])
            offset += durations[i]

        # Compute loss
        if target is not None:
            loss = torch.nn.functional.mse_loss(torch.cat(preds), target)
            loss = loss / target.shape[0] # Normalize by number of frames
            return preds, loss
        
        return preds
