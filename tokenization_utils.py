from typing import Dict, Optional, List
from collections import OrderedDict

from tokenization_utils_base import PreTrainedTokenizerBase




class Trie:
    
    
    def __init__(self):
        self.data = {}
        self._tokens = set()
        
        
    
                        
class PreTrainedTokenizer(PreTrainedTokenizerBase):
    
    
    
    
    def __init__(self, **kwargs):
        
        
        self.tokens_trie = Trie()
        
        
        self.added_tokens_decoder.update(kwargs.pop("added_tokens_decoder", {}))
        self._added_tokens_encoder: Dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}
        
        
        
        
        super().__init__(**kwargs)
        
        
        self._add_tokens(
            [token for token in self.all_special_tokens_extended if token not in self._added_tokens_encoder],
            special_tokens = True,
        )
        
        
        self._decoder_use_source_tokenizer = False