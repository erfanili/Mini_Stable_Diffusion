
import torch
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict, Any







from unet_2d_blocks import get_down_block, get_up_block, get_mid_block



from configuration_utils import register_to_config




















































#69
class UNet2DConditionModel():
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #168
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        flip_sin_to_cos:bool = True,
        freq_shift: int =0,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2d",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UnetMidBlock2DcrossAttn",
        up_block_types: tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBloc2D"),
        only_cross_attention: Union[bool,Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320,640,1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 128,
        transformer_laers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
    ):
        
        
        super().__init__()
        
        self.sample_size = sample_size 
        
        
        
        
        num_attention_heads = num_attention_heads or attention_head_dim
        
        
        
        
        
        
        
        
        
        
        
        
        
        #348
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i==len(block_out_channels) - 1
            
            down_block = get_down_block(
                in_channels = input_channel,
                out_channels = output_channel,
            )
            self.down_blocks.append(down_block)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    #383
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels = block_out_channels[-1],
            renet_eps = norm_eps,
            resnet_act_fn = act_fn
        )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i ==len(block_out_channels) -1
            
            
            
            
            
            
            
            
            
            
            
            
            up_block = get_up_block(
                up_block_type,
            )
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        #1031
        def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask : Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ):
            
            
            
            
            #1098
            forward_upsample_size = False
            upsample_size = None
            
            if attention_mask is not None:
                
                attention_mask = (1- attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)
                
            if encoder_attention_mask is not None:
                encoder_attention_mask  = (1- encoder_attention_mask.to(sample.dtype)) * 10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            
            #1133
            t_emb = self.get_time_embed(sample = sample, timestep = timestep)
            emb = self.time_embedding(t_emb, timestep_cond)
            
            
            class_emb = self.get_class_embed(sample = sample, class_labels = class_labels)
            
            
            
            
            #1156
            encoder_hidden_states = self.process_encoder_hidden_states(
                encoder_hidden_states = encoder_hidden_states, added_cond_kwargs = added_cond_kwargs
            )
        
        
        
            sample = self.conv_in(sample)
        
            down_block_res_sample = (sample,)
            
            #1201
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                            
                    sample, res_samples = downsample_block(
                        hidden_states = sample,
                        temb = emb,
                        encoder_hidden_states = encoder_hidden_states,
                        attention_mask = attention_mask,
                        cross_attention_kwargs = cross_attention_kwargs,
                        encoder_attention_mask = encoder_attention_mask,
                        
                    )

                else:
                    sample, res_samples = downsample_block(hidden_states = sample, tempb = emb)
                    
                down_block_res_samples += res_samples
                
                
            #1236
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states = encoder_hidden_states,
                        attention_ask = attention_mask,
                        cross_attention_kwargs = cross_attention_kwargs,
                        encoder_attention_mask = encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)
                    
            
            #1261
            for i, upsample_block in enumerate(self.upblocks):
                is_final_block = i == len(self.up_blocks) -1
                
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
                
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]
                    
                    
                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states = sample,
                        temb = emb,
                        res_hidden_states_tuple = res_samples,
                        encoder_hidden_staes =encoder_hidden_states,
                        cross_attention_kwargs = cross_attention_kwargs,
                        upsample_size = upsample_size,
                        attention_mask =attention_mask,
                        encoder_attention_mask = encoder_attention_mask,
                    )
        
                else:
                    sample = upsample_block(
                        hidden_states = sample,
                        temb = emb,
                        res_hidden_states_tuple = res_samples,
                        upsample_suze = upsample_size,
                        
                    )   
                    
                    
            #1292
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)
            
            
            
            return (sample,)  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        output_channel = block_out_channels[0]
        
        
        
        
        
        
        
        
        
        