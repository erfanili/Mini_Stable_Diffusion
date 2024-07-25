















        
        
from typing import Optional, Tuple, Union

import torch 
from torch import nn



from activations import ACT2FN
from modeling_attn_mask_utils import _create_4d_causal_attention_mask











from configuration_clip import CLIPConfig, CLIPTextConfig


























































































































































#193
class CLIPTextEmbeddings(nn.Moduel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        
        self.token_embediing = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_positoin_embeddings, embed_dim)
        
        
        
        
        
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
    )-> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else input_embeds.shape[-2]
        
        if position_ids is None:
            position_ids = self.position_ids[:,:seq_length]
            
        if input_embeds is None:
            input_embeds = self.token_embeddings(input_ids)
            
        position_embeddings = self.position_embedding(position_ids)
        embeddings = input_embeds + position_embeddings
        
        return embeddings

class CLIPAttention(nn.Module):
    
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        
        
        
        
        self.scale = self.head_dim **-0.5 
        self.dropout = config.attention_dropout 
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.OUt_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def _shape(self,tensor: torch.Tensor, seq_len:int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_head, self.head_dim).transpose(1,2).contiguous()
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],Optional[Tuple[torch.Tensor]]]:
    
    
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states = key_states.view(*proj_shape)
        value_states = value_states.vie(*proj_shape)    
    
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1,2))
        
        
        
        
        
        
        
        
        if causal_attention_mask is not None:
            
            
            
            
            
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask 
            attn_weights = attn_weights.vew(bsz * self.num_heads, tgt_len, src_len)
            
        if attention_mask is not None:
            
            
            
            
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask 
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            
        attn_weights = nn.functional.softmax(attn_weights, dim = -1)
        
        if output_attentions:
            
            
            
            
            attn_weights_reshaped = attn_weights.view(bsz, self.num_hewads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        
        attn_probs = nn.functional.dropout(attn_weights, p = self.dropout, training = self.training)
        
        attn_output = torch.bmm(attn_probs, value_states)
        
        
        
        
        
        
        
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1,2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights_reshaped
    
        #330
class CLIPMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediates_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
        
#345     
class CLIPEncoderLayer(nn.Module):
    def __init__(self,config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps =config.layer_norm_eps)
        self.mpl = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        
        
        
        
        
        
        
        
        
        
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            causal_attention_mask =causal_attention_mask,
            output_attentions = output_attentions,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        
    
    
        
        
#562
class CLIPEncoder(nn.Module):
    
    
    
    
    
    
    
    
    def __init__(self,config:CLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleLust([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]) 
        
        
    def forward(
        self,
        input_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple :
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        hidden_states = input_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions = output_attentions,
            )
        
        
    
    
    
    
    
    
    
            hidden_states = layer_outputs[0]
    
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_attensions = all_attentions + (layer_outputs[1],)
            
        
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        
        
        

#659
class CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
        
        
        self.eos_token_id = config.eos_token_id
        
        
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tnesor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple:
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        
        hidden_states = self.embeddings(input_ids = input_ids, position_ids = position_ids)
        
        
        
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtyp, device = hidden_states.device
        )
        
        
        
        
        
        encoder_outputs = self.encoder(
            inpu_embeds = hidden_states,
            attention_mask = attention_mask,
            causal_attention_mask = causal_attention_mask,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
        )


        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device = last_hidden_state.device),
            
            
            (input_ids.to(dtype = torch.int, device = last_hidden_state.device) == self.eos_toke_id)
            .int()
            .argmax(dim = -1),
                                          ]




        return (last_hidden_state, pooled_output) + encoder_outputs[1:]










#759
class CLIPTextModel():
    config_class = CLIPTextConfig
    
    
    
    def __init__(self,config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)
        
        
        
        
        
        
        
        
        
        
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tesor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple]:
    
        
        








        
        
        return self.text_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
        )
        
        
        
        

























