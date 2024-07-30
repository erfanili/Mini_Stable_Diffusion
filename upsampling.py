from typing import Optional, Tuple, Any, List, Dict 
import torch
from torch import nn 
from normalization import RMSNorm

class Upsample2D(nn.Module):
    
    def __init__(
        self,
        channels: int, 
        use_conv: bool = False, 
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        padding = 1,
        eps = None, 
        bias = True,
        norm_type = None,
        elementwise_affine = None,
        name: str = "conv",
        interpolate = True,
    ):
        
        super().__init__()
        self.channels = channels 
        self.out_channels = out_channels or channels 
        self.use_conv_transpose = use_conv_transpose 
        self.name = name 
        self.interpolate = interpolate
        
        
        
        if norm_type == "ln_norm":
            self.norm == nn.LayerNorm(channels, eps, elementwise_affine) 
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else: raise ValueError(f"unknown norm_type: {norm_type}")
        
        
        conv = None 
        
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4 
                conv = nn.ConvTranspose2d(
                    channels, self.out_channels, kernel_size = kernel_size, stride = 2, padding = padding, bias = bias
                )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size = kernel_size, padding = padding, bias = bias) 
        if name == "conve":
            self.conv = conv 
        else:
            self.Conv2d_0 = conv 
    
    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor: 
        if self.norm is not None:
            hidden_states  = self.norm(hidden_states.permute(0,2,3,1)).permute(0,3,1,2)
            
            
        if self.use_conv_transpose:
            return self.conv(hidden_states) 
            
        if self.use_conv:
            if self.name =="conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
                
        return hidden_states         
            
            
                