import torch 
from torch import nn 
from typing import Dict, List, Optional, Tuple , Any
from activations import GEGLU
from torch.utils import maybe_allow_in_graph
from embeddings import SinusoidalPositionalEmbedding
from normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from attention_processor import Attention


@maybe_allow_in_graph 
class BasicTransformerBlock(nn.Module):
    
    
    def __init__(
        self, 
        dim:int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn:str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False, 
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None, 
        num_positional_embeddings: Optional[int] = None, 
        ada_norm_continuous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None, 
        ff_inner_dim: Optional[int] = None, 
        ff_bias: bool = True, 
        attention_out_bias: bool = True,
    ):
        super().__init__() 
        self.only_cross_attention = only_cross_attention 
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero" 
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm" 
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single" 
        self.use_laye_norm = norm_type == "layer_norm" 
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous" 
        
        self.norm_type = norm_type 
        self.num_embeds_ada_norm = num_embeds_ada_norm 
        
        
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length = num_positional_embeddings) 
        else: 
            self.pos_embed = None 
            
            
        if norm_type == "ada_norm": 
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) 
        elif norm_type == "ada_norm_zer": 
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm) 
        elif norm_type == "ada_norm_continuous": 
            self.norm1 = AdaLayerNormContinuous(
                dim, 
                ada_norm_continuous_conditioning_embedding_dim,
                norm_elementwise_affine, 
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else: 
            self.norm1 = nn.LayerNorm(dim, elementwise_affine = norm_elementwise_affine, eps = norm_eps) 
            
        self.attn1 = Attention(
            query_dim = dim,
            heads = num_attention_heads, 
            dim_head = attention_head_dim, 
            dropout = dropout, 
            bias = attention_bias, 
            cross_attention_dim = cross_attention_dim if only_cross_attention else None, 
            upcast_attention = upcast_attention,
            out_bias = attention_out_bias
        )
        
        if cross_attention_dim is not None:
            
            
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) 
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continuous_conditioning_embedding_dim,
                    norm_elementwise_affine, 
                    norm_eps, 
                    ada_norm_bias,
                    "rms_norm",
                )
                
            else: 
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
            
            self.attn2 = Attention(
                query_dim = dim,
                cross_attention_dim = None, 
                heads = num_attention_heads, 
                dim_head = attention_head_dim, 
                dropout = dropout, 
                bias = attention_bias, 
                upcast_attention = upcast_attention, 
                out_bias = attention_out_bias
            )   
            
        else:
            self.norm2 = None
            self.attn2 = None 
        
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continuous_conditioning_embedding_dim, 
                norm_elementwise_affine, 
                norm_eps, 
                ada_norm_bias, 
                "layer_norm"
            )
        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine) 
        elif norm_type == "layer_norm_i2vgen":
        
            self.norm3 = None 
            
        self.ff = FeedForward(
            dim,dropout = dropout,
            activation_fn = activation_fn,
            final_dropout = final_dropout,
            inner_dim = ff_inner_dim,
            bias = ff_bias,
        )    
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
        encoder_hidden_states: Optional[torch.Tensor] = None, 
        encoder_attention_mask: Optional[torch.Tensor]= None,
        timestep: Optional[torch.LonegTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None, 
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        batch_size = hidden_states.shpae[0] 
        if self.notm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep) 
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype = hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            
            
            
        
        if self.pos_embed is not None:
            norm_hidden_states = self.pose_embed(norm_hidden_states) 
            
            
        cross_attetnion_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attetnion_kwargs.pop("gligen", None)
        
        attn_output = self.attn1(
            norm_hidden_states, 
            encoder_hidden_states = encoder_hidden_states if self.only_cross_attetnion else None, 
            attention_mask = attention_mask, 
            **cross_attention_kwargs,
        )
        
        hidden_states = attn_output + hidden_states 
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        
        
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_state = self.norm2(hidden_states, timestep)
                
                
            attn_output = self.attn2(
                norm_hidden_states, 
                encoder_hidden_states= encoder_hidden_states, 
                attentioN_mask = encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
            
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        
        ff_output = self.ff(norm_hidden_states)
        
        hidden_states = ff_output + hidden_states 
        if hidden_states.ndim ==4:
            hidden_states = hidden_states.squeeze(1)
        
        return hidden_states
    
    
    
class FeedForward(nn.Module):
    
    def __init__(
        self, 
        dim: int, 
        dim_out: Optional[int] = None,
        mult: int = 4, 
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False, 
        inner_dim = None, 
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim 
        
        if activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, approximate = "tanh", bias = bias)
            
        
        self.net = nn.ModuleList([])
        self.net.append(act_fn) 
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim,dim_out, biad = bias))
        
        if final_dropout:
            self.net_append(nn.Dropout(dropout))
            
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states