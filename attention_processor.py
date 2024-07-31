import torch 
from typing import Optional, Dict, List, Any, Tuple
from torch import nn 
import torch.nn.functional as F 


import inspect

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None, 
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False, 
        out_dim: int = None,
        out_bias: bool = True,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads 
        self.query_dim = query_dim 
        self.use_bias = bias  
        self.is_cross_attention = cross_attention_dim is not None 
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim 
        self.dropout = dropout 
        self.fused_projections = False 
        
        
        self.heads = out_dim // dim_head if out_dim is not None else heads 
        
        
        
        
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias = bias) 
        
        if not self.only_cross_attention:
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias = bias) 
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias = bias) 
            
        else: 
            self.to_k = None ,
            self.to_v = None, 
            
            
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias = out_bias))
        self.to_out.append(nn.Dropout(dropout)) 
        
        if processor is None: 
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor() 
            )
        self.set_processor(processor)
        
            
            
            
    def set_processor(self, processor: "AttnProcessor") -> None:
        
        
        
        self.processor = processor 
        
        
    def forward(
        self,
        hidden_states: torch.Tensor, 
        encoder_hidden_states: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        
        cross_attentions_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
        
        return self.processor(
            self, 
            hidden_states, 
            encoder_hidden_states = encoder_hidden_states, 
            attention_mask = attention_mask, 
            **cross_attention_kwargs,
        )
        
        
class AttnProcessor:
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor, 
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states 
        
        
        input_ndim = hidden_states.ndim 
        
        if input_ndim == 4:
            batch_size, channel, height, height, width = hidden_states.shape 
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1,2) 
            
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states) 
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states 
            
            
            
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask) 
        hidden_states = torch.bmm(attention_probs, value) 
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        
        hidden_states = attn.to_out[0](hidden_states)
        hiddeN_states = attn.to_out[1](hidden_states)
        
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1,-2).reshape(batch_size, channel, height, width)
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual 
            
        hidden_states = hidden_states / attn.rescale_output_facctor 
        
        
        return hidden_states 
    
    
    
    

class AttnProcessor2_0:
    
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
    def __call__(self,
                 attn: Attention,
                 hidden_states: torch.Tensor,
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None, 
                 temb: Optional[torch.Tensor] = None, 
                 *args,
                 **kwargs,
                 ) -> torch.Tensor: 
        
        residual = hidden_states 
        input_ndim = hidden_states.ndim 
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose( 1,2) 
            
            
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            
            
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1,2)).transpose(1,2)
            
            
        query = attn.to_q(hidden_states)
        
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
            
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) 
        
        inner_dim = key.shape[-1] 
        head_dim = inner_dim // attn.heads 
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1,2) 
        
        
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1,2) 
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1,2)
        
        hidden_states = F.scaled_dot_product_attention(
            query,key,value, attn_mask = attention_mask, dropout_p = 0.0, is_causal = False
        )
        
        hidden_states = hidden_states.transpose(1,2).reshape(batch_size, -1, attn.heads * head_dim) 
        hidden_states = hidden_states.to(query.dtype) 
        
        hidden_states = attn.to_out[0](hidden_states) 
        hidden_states = attn.to_out[1](hidden_states) 
        
        if input_ndim ==4:
            hidden_states = hidden_states.transpose(-1,-1).reshape(batch_size, channel, height, width)
            
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual 
            
            
        hidden_sates = hidden_states / attn.rescale_output_factor
        
        
        return hidden_states 