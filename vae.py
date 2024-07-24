













from typing import Optional, Tuple


import torch 
import torch.nn as nn




























class Encoder(nn.Module):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        norm_num_groups: int = 32,
        layers_per_block: int = 32,
        act_fn = "silu",
        double_z: bool = True,
    ):
        
        
        
        super().__init__()
        self.layers_per_block = layers_per_block
        
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size = 3,
            stride = 1,
            padding =1,
        )
        
        self.down_blocks = nn.ModuleList([])
        
        output_channel = block_out_channels[0]
        for i , down_block_type in enumerate(down_block_types):
            input_channel = output_channeloutput_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) -1
            
            down_block = get_down_block(
                down_block_type,
                num_layers = self.layers_per_block,
                in_channels = input_channel,
                out_channels = output_channel,
                resnet_eps = 1e-6,
                
            )
            self.down_blocks.append(down_block)
            
            
            
            
            
            
            
        self.mid_block = UnetMidBlock2d(
            in_channels = block_out_channels[-1],
            resnet_eps = 1e-6,
            resnet_cat_fn = act_fn,
        )








        self.conv_norm_out = nn.GroupNorm(num_channels = block_out_channels[-1], num_groups = norm_num_groups, eps = 1e-6)
        self.conv_act = nn.SiLU()
        
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3 , padding = 1)



    def forward(self, sample:torch.Tensor) -> torch.Tensor:
        
        
        sample = self.conv_in(sample)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        for down_block in self.down_blocks:
            sample = down_block(sample)
        
        
        sample = self.mid_block(sample)
    






    
class Decoder(nn.Module):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpBDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",
        mid_block_add_attention = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )
        
        
        self.up_blocks = nn.ModuleList([])
        
        
        
        
        self.mid_block = UnetMidBlock2d(
            in_channel = block_out_channels[-1],
            resnet_eps = 1e-6,
            resnet_act_fn = act_fn,
            attention_head_dim = block_out_channels[-1],
            resnet_groups = norm_num_groups,
            add_attention = mid_block_add_attention,
        )





        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i , up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            
            is_final_block = i ==len(block_out_channels) -1 
            
            up_block = get_up_block(
                up_block_type,
                num_layers = self.layers_per_block + 1,
                in_channels = prev_output_channel,
                out_channels = output_channel,
                prev_output_channel = None,
                resnet_eps = 1e-6,
                resnet_act_fn = act_fn,
                resnet_groups = norm_num_groups,
            )
            
            
            
            
        self.up_blocks.append(up_block)
        prev_output_channel = output_channel
        
        
        
        
        
        self.conv_norm_out = nn.GroupNorm(num_channels = block_out_channels[0], num_groups = norm_num_groups, eps = 1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3 , padding = 1)
            
    
    
    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        
        sample = self.conv_in(sample)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        sample = self.mid_block(sample,latent_embeds)
            
            
            
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
        
        
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conve_norm_out(sample, latent_embeds)                
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        
        return sample    
        
            
            
            
            
            
            