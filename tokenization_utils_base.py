from typing import Union, Dict, Optional, List
import copy
import os 
import json

from matplotlib.font_manager import json_load

class AddedToken:
    
    
    def __init__(
        self, content: str, single_word = False, lstrip = False, rstrip = False, special = False, normalized = None,
    ):
        self.content = content
        self.single_word = single_word 
        self.lstrip = lstrip 
        self.rstrip = rstrip 
        self.special = special 
        self.normalized = normalized  if normalized is not None else not special 
        
    def __getitem__(self):
        return self.__dict__ 
    
    def __str__(self):
        return self.content 
    
    


class PreTrainedTokenizerBase():
    
    
    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str,str]= {}
    _auto_class: Optional[str] = None
    
    
    
    def __init__(self, **kwargs):
        
        self.init_inputs = ()
        self.init_kwargs = copy.deepcopy(kwargs)
        self.name_or_path = kwargs.pop("name_or_path","")
        self._processor_class = kwargs.pop("processor_class", None)
        
        
        
        
        
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str,os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        token: Optional[Union[str,bool]] = None,
        trust_remote_code = False,
        local_files_only: bool = False,
        **kwargs
    ):
        resume_download = kwargs.pop("resume_download",None)
        commit_hash = kwargs.pop("_commit_hash",None)
        init_configuration = {}
        
        resolved_vocab_files = []
        
        is_local = os.path.isdir(pretrained_model_name_or_path)
        
        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            token=token,
            cache_dir = cache_dir,
            locak_files_only = local_files_only,
            _commit_hash = commit_hash,
            _is_local = is_local,
            trust_remote_code = trust_remote_code,
            **kwargs
        )
        
    
    @classmethod
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        token = None,
        cache_dir = None,
        local_files_only = False,
        _commit_hash = None,
        _is_local = False,
        trust_remote_code = False,
        **kwargs,
    ):
        
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        
        with open(tokenizer_config_file, encoding = "utf-8") as tokenizer_config_handle:
            init_kwargs = json_load(tokenizer_config_handle)
            
        config_tokenizer_class = init_kwargs.get("tokenizer_class")
        init_kwargs.pop("tokenizer_class", None)
        
            
            
        tokenizer = cls(*init_inputs, **init_kwargs)
        
        
        return tokenizer