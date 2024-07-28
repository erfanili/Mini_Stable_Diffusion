import torch
from torch import nn
from typing import Optional  
     
        #188 
class ResnetBlock2D(nn.Module):
    
    def __init__(
        self,
        *, 
        in_channels: int, 
        out_channels: Optional[int] = None, 
        conv_chortcut: bool = False, 
        dropout: float = 0.0,
        temb_channels: int = 512, 
        groups: int = 32, 
        groups_out: Optional[int] = None, 
        up: bool= False, 
        down: bool= False,
        
    ):
        super().__init__()
        
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.up = up 
        self.down = down 
        
        
        
        
        
        
    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        
        hidden_states  = input_tensor
        hidden_states = self.norm1(hidden_states) 
        hidden_states = self.nonlinearity(hidden_states) 
        
        
        hidden_states = self.conv1(hidden_states)
        
        
        if self.time_emb_proj is not None:
            if not self.skip_time_act: 
                temb = self.nonlinearity(temb) 
            temb = self.time_emb_proj(temb)[:,:, None, None]
            
            
            
        hidden_states = self.nonlinearity(hidden_states) 
        
        hidden_states = self.dropout(hidden_states) 
        hidden_states = self.conv2(hidden_states) 
        
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor 
        
        
        return output_tensor 