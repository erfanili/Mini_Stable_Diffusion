import re
import json 

from tokenization_utils_base import AddedToken
from tokenization_utils import PreTrainedTokenizer


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json", "merges_file": "merges.txt"
}


class CLIPTokenizer(PreTrainedTokenizer):
    
    vocab_dfiles_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors = "replace",
                 unk_token = "<|endoftext|>",
                 bos_token = "<|startoftext|>",
                 eos_token = "<endoottext|>",
                 pad_token = "<|endoftext|>",
                 ):
        bos_token = AddedToken(bos_token, lstrip = False, rstrip = False) if isinstance(bos_token, str) else bos_token 
        eos_token = AddedToken(eos_token , lstrip = False, rstrip = False) if isinstance(eos_token, str) else eos_token 
        unk_token = AddedToken(unk_token, lstrip = False, rstrip = False) if isinstance(unk_token, str) else unk_token
        
        
        with open(vocab_file, encoding = "utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v:k for k,v in self.encoder.items()}
        with open(merges_file, encoding = "utf-8") as merges_handle:
            bpe_merges = merges_handle.read().strip().split("\n")[1 : 49152 - 256 - 2+1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {"<|startoftext \|": "<|startoftext|>", "<endoftext|>": "<|endoftext|>"}
        
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )