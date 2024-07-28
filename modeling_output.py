

from dataclasses import dataclass 
import torch 
from typing import Tuple, Optional

@dataclass 
class BaseModelOutputWithPooling(ModelOutput):
    
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]]= None
    attentions = Optional[Tuple[torch.FloatTensor,...]] = None 