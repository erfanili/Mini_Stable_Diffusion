import torch 

from typing import Union, Tuple, Optional, List



class Attention_mask_converter():
    
    is_causal: bool
    sliding_window: int 
    
    def __init__(self,is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

def _create_4d_causal_attention_mask(
    input_shape: Union[torch.Size, Tuple, List],
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    
    attn_mask_converter = Attention_mask_converter(is_causal = True, sliding_window = sliding_window)
    
    
    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, device =device 
    )