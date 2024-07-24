












from typing import Tuple, Union, Optional

import torch 
import torch.nn as nn














from vae import Decoder, DiagonalGaussianDistribution, Encoder





class AutoencoderKL():
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        sample_size: int = 32,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__()
        
        
        
        
        
        
        
        self.encoder = Encoder(
            in_channels = in_channels,
            out_channels = latent_channels,
            down_block_types = down_block_types,
            block_out_channels = block_out_channels,
            layers_per_block = layers_per_block,
            act_fn = act_fn,
        )
        
        
        
        
        self.decoder = Decoder(
            in_channels = latent_channels,
            out_channels = out_channels,
            up_block_types = up_block_types,
            block_out_channels = block_out_channels,
            layers_per_block = layers_per_block,
            act_fn = act_fn,
        )
        
        self.quant_conv = nn.Conv2d(2*latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.pose_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        def encode(
            self, x: torch.Tensor
        ):
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            h = self.encoder(x)
            
            if self.quant_conv is not None:
                moments = self.quant_conv(h)
            else:
                moments = h
            
            posterior = DiagonalGaussianDistribution(moments)
            

            
            
            return posterior
    
    def _decode(self,z:torch.Tensor) -> torch.Tensor:
        
        
        
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        
        dec = self.decoder(z)
        
        
        
        
        return dec
         
        
    def decode(
        self, z: torch.FloatTensor
    ) -> torch.FloatTensor:
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        decoded = self._decode(z).sample
        
        
        
        
        return decoded
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        def forward(
            self,
            sample: torch.Tensor,
        ) -> torch.Tensor:
            
            
            
            
            
            
            
            
            
            
            
            x = sample
            posterior = self.encode(x).latent_dist
            
            
            
            
            z = posterior.mode()
            dec = self.decode(z).sample
            
            
            
            return dec
            
        
        
        
        
        
        
        
        
        