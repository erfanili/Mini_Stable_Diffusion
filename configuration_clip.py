

















from typing import Optional, Tuple, Union

import torch


from torch import nn








#33
class CLIPTextConfig():
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    model_type = "clip_text_model"
    
    def __init__(
        self,
        vocab_size = 49408,
        hidden_size = 512,
        intermediate_size = 2048,
        projection_dim = 512,
        num_hidden_layers = 12,
        num_attention_heads = 8,
        max_position_embeddings = 77,
        hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5,
        attention_dropout = 0.0,
        initializer_range = 0.02,
        initializer_factor = 1.0,
        
        
        pad_token_id = 1,
        bos_token_id = 494406,
        eos_token_id = 494407,
        **kwargs,
    ):
        super().__init__(pad_token_id = pad_token_id, bos_token_is = bos_token_id, eos_token_id = eos_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range 
        self.initializer_factor = initializer_factor 
        self.atention_dropout = attention_dropout 
        
        
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs)  -> PretrainedConfig:
        

        config_dict = cls.get_config_dict(pretrained_model_name_or_path)
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["text_config"]
            

        
        
        
        
        
        
        
        return cls.from_dict(config_dict)
        













































































































#260
class CLIPConfig(PretrainedConfig):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    model_type = "clip"
    
    def __init__(
        self,text_config = None, vision_config = None, projection_dim = 512, logit_scale_init_value = 2.6592, **kwargs
    ):
        
        
        
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        
        super().__init__(**kwargs)
        
        
        
        if text_config_dict is not None:
            if text_config is None:
                text_config ={}
                
                
            
            _text_config_dict = CLIPTextConfig(**text_config_dict).to_dict()
            
            

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        if text_config is None:
            text_config = {}
            
        if vision_config is None:
            vision_config = {}
    
    
    

        self.text_config = CLIPTextConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)
        
        self.projection_dim = projection_dim
    
    
    
    
    
    
    
    
    
    







##line number + 500




