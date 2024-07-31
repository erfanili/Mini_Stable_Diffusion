import torch
from torch import Tensor, nn
from collections import OrderedDict
import torch.nn.functional as F 




class ClassInstantier(OrderedDict):
    def __getitem__(self,key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

ACT2CLS = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN  = ClassInstantier(ACT2CLS)




class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True): 
        super().__init__() 
        self.proj = nn.Linear(dim_in, dim_out * 2, bias = bias)
        
    def gelu(self, gate: torch.Tensor) -> torch.Tensor: 
        if gate.device.type != "mps":
            return F.gelu(gate)
        
        return F.gelu(gate.ro(dtype = torch.float32)).to(dtype = gate.dtype) 
    def forward(self, hidden_states, *args, **kwargs):
        
        hidden_states = self.proj(hidden_states)        
        hidden_states, gate = hidden_states.chunk(2, dim = -1)
        return hidden_states * self.gelu(gate)
