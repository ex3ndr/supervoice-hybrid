import torch
from .transformer import Transformer

class SupervoceHybridStage1(torch.nn.Module):
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
        self.audio_embedding = torch.nn.ModuleList([torch.nn.Embedding(1024, self.n_dim) for _ in range(8)])
        for embedding in self.audio_embedding:
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

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
        
        # Input projection
        self.input_projection = torch.nn.Linear(self.n_dim * 2, self.n_dim, bias=False)


    def forward(self, *, condition_text, condition_audio, target = None, duration = None):
        
        # Check batch size
        B = len(condition_text)
        assert len(condition_audio) == B

        # Durations
        if duration is not None:
            assert target is None
            assert len(duration) == B
        else:
            assert target is not None
            assert len(target) == B
            duration = [t.shape[0] for t in target]

        # Concatenate inputs
        inputs_text = []
        inputs_audio = []
        inputs_positional = []
        for i in range(B):
            d = duration[i]
            inputs_text.append(torch.pad(condition_text[i] + 1, (0, condition_text[i].shape[0] - d), "constant", 0))
            inputs_audio.append(torch.pad(condition_audio[i][0] + 1, (0, condition_audio[i][0].shape[0] - d), "constant", 0))
            inputs_positional.append(torch.arange(d).to(t.device, non_blocking=True))
        inputs_text = torch.cat(inputs_text)
        inputs_audio = torch.cat(inputs_audio)
        inputs_positional = torch.cat(inputs_positional)
        inputs_text += self.positional_embedding_text(inputs_positional)
        inputs_audio += self.positional_embedding_audio(inputs_positional)
        inputs = torch.stack([inputs_text, inputs_audio], dim=1)
        inputs = self.input_projection(inputs)

        # Run transformer
        x = self.transformer(inputs)

            # t_x = self.text_embedding(torch.pad(condition_text[i] + 1, (0, condition_text[i].shape[0] - d), "constant", 0))
            # t_x = t_x + self.positional_embedding_text(torch.arange(d).to(t.device, non_blocking=True))

            # # Audio embedding
            # a_x = self.audio_embedding[0](torch.pad(condition_audio[i][0] + 1, (0, condition_audio[i][0].shape[0] - d), "constant", 0))
            # # for j in range(1, 8):
            # #     a_x = a_x + self.audio_embedding[j](torch.pad(condition_audio[i][j] + 1, (0, condition_audio[i][j].shape[0] - d), "constant", 0))
            # a_x = a_x + self.positional_embedding_audio(torch.arange(d).to(t.device, non_blocking=True))

            # # Condition
            # cond = t_x + a_x

            # # Append
            # inputs.append(cond)

        # Run transformer
        # x = self.transformer(inputs)

        return x