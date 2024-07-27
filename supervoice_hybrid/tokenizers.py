import torch
import sentencepiece as spm
from seamless_communication.models.unity.char_tokenizer import load_unity_char_tokenizer

class SentencePieceTextTokenizer:
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)

    def encode(self, text):
        return torch.tensor(self.sp.encode(text), dtype=torch.long).squeeze(0).squeeze(0)

    def encode_sample(self, text):
        return torch.tensor(self.sp.encode(text, enable_sampling=True, alpha=0.1, nbest_size=-1), dtype=torch.long).squeeze(0).squeeze(0)


class UnitTextTokenizer:
    def __init__(self):
        text_tokenizer = load_unity_char_tokenizer("nar_t2u_aligner")
        self.tokenizer = text_tokenizer.create_raw_encoder()
        self.vocab_info = text_tokenizer.vocab_info
        self.bos_idx = self.vocab_info.bos_idx
        self.eos_idx = self.vocab_info.eos_idx
        self.pad_idx = self.vocab_info.pad_idx

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text)

    def encode_sample(self, text: str) -> torch.Tensor:
        return self.tokenizer(text)