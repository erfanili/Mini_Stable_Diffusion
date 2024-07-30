from typing import Optional, Dict, Any, List
from configuration_utils import register_to_config
import torch 
from torch import nn
import torch.nn.functional as F
from attention import BasicTransformerBlock


class Transformer2DModel():
    
    
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None, 
        patch_size: Optional[int] = None,
        num_embeds_ada_norm: Optional[int] = None, 
        use_linear_projection: bool = False,
        norm_type: str = "layer_norm",
        caption_channels: int = None, 
        interpolation_scale: float = None,
    ):
        super().__init__() 
        
        self.use_linear_projection = use_linear_projection 
        self.interpolation_scale = interpolation_scale 
        self.caption_channels = caption_channels 
        self.num_attention_heads = num_attention_heads 
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim 
        self.in_channels = in_channels 
        self.out_channels = in_channels if out_channels is None else out_channels 
        
        
        
        
        
    def _init_continuous_input(self, norm_type):
        self.norm = torch.nn.GroupNorm(
            num_groups = self.config.norm_num_groups, num_channels = self.in_channels, eps = 1e-6, affine = True
        )
        if self.use_linear_projection:
            self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim) 
        else:
            self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim, kernel_size = 1, stride = 1, padding = 0)
        
        
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.num_attention_heads,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        
        if self.use_linear_projection:
            self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels) 
        else: 
            self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels, kernel_size = 1, stride = 1, padding = 0)
            


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None, 
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        
        if attention_mask is not None and attention_mask.ndim == 2:
            
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0 
            attention_mask = attention_mask.unsqueeze(1)
            
        if encoder_attention_mask is not None and encoder_attention_mask.ndim ==2:
            encoder_attention_mask = (1- encoder_attention_mask.to(hidden_states.dtyp)) * -10000 
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1) 
        
        if self.is_input_continuous:
            batch_size, _, height, width = hidden_states.shape
            residual = hidden_states 
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        
        if self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size 
            hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep
            )
            
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask = attention_mask,
                encoder_hidden_states = encoder_hidden_states,
                encoder_attention_mask = encoder_attention_mask,
                timestep = timestep,
                cross_attention_kwargs = cross_attention_kwargs,
                class_labels = class_labels,
                
            )
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states = hidden_states,
                residual = residual,
                bach_size = batch_size,
                height = height,
                width = width,
                inner_dim = inner_dim,
            )
            
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(
                hidden_states = hidden_states,
                timestep = timestep,
                class_labels = class_labels,
                embedded_timestep = embedded_timestep,
                height = height,
                width = width,
            )
            return (output,)
        
        
        
    def _operate_on_continuous_inputs(self, hidden_states):
        batch, _, height, width = hidden_states.shape 
        hidden_states = self.norm(hidden_states) 
        
        
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states) 
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0,2,3,1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1] 
            hidden_states = hidden_states.permute(0,2,3,1).reshape(batch, height * width, inner_dim) 
            hidden_states = self.proj_in(hidden_states) 
            
        return hidden_states, inner_dim 
    
    
    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs):
        batch_size = hidden_states.shape[0] 
        hidden_states = self.pos_embed(hidden_states)
        embedded_timestep = None 
        
        if self.adaln_single is not None:
            
            timestep , embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size = batch_size, hidden_dtype = hidden_states.dtype
            )
        if self.caption_projection is not None:
            encoder_hidden_statse = self.caption_projection(encoder_hidden_states) 
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            
            return hidden_states, encoder_hidden_states, timestep, embedded_timestep 
        
    
    def _get_output_for_continuous_inputs(self, hidden_states, residual, batch_size, height, width, inner_dim):
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, inner_dim).permute(0,3,1,2).contiguous()
            )
            
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, inner_dim).permute(0,3,1,2).contiguous()
            )
            output = hidden_states + residual
            return output 
        
    def _get_output_for_patched_inputs(
        self,hidden_states, timestep, class_labels, embedded_timestep, height = None, width = None
    ):
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype = hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim =1)
            hidden_states = self.norm_out(hidden_states) * (1+ scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.nofig.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[: None]).chunk(2, dim =1)
            hidden_states = self.norm_out(hidden_states)
            
            hidden_states = hidden_states * (1+ scale) + shift 
            hidden_states = self.proj_out(hidden_states) 
            hidden_states = hidden_states.squeeze(1)
            
            
            
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape = (-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape = (-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            return output
        
        
    
        