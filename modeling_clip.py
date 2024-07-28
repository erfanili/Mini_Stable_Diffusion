















        
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch 
from torch import nn



from activations import ACT2FN
from modeling_attn_mask_utils import _create_4d_causal_attention_mask


from modeling_output import BaseModelOutputWithPooling








from configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig























#63
@dataclass
class CLIPVisionModelOutput():
    
    
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None






















































































#158
class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config:CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.hidden_size
        self.patch_size = config.patch_size
        
        
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            strid = self.patch_size,
            bias = False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patchers + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype 
        patch_embeds = self.patch_embedding(pixel_values.to(dtype = target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1,2)
        
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim = 1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings





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
        
        
        
#816
class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size 
        
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)        


        def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
        )->Union[Tuple, BaseModelOutputWithPooling]:
            
            
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions 
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            
            hidden_state = self.embeddings(pixel_values)
            hidden_states = self.pre_layernorm(hidden_states)
            
            encoder_outputs = self.encoder(
                inputs_embeds =hidden_states ,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
            )
            
            last_hidden_state = encoder_outputs[0]
            pooled_output = last_hidden_state[:,0,:]
            pooled_output = self.post_layernorm(pooled_output)
            
            
            return BaseModelOutputWithPooling(
                last_hidden_states = last_hidden_state, 
                pooler_output = pooled_output,
                hidden_states = encoder_outputs.hidden_states,
                attentions = encoder_outputs.attentions,
            )








#1247
class CLIPVisionModelWithProjection():
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        
        self.vision_model = CLIPVisionTransformer(config)
        
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, biad = False)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None
    ):
        #1296
        vision_outputs = self.vision_model(
            pixel_values = pixel_values,
            output_attentions = output_attentions,
            output_hidden_statse = output_hidden_states
        )
        
        
        pooled_output = vision_outputs[1]
        
        image_embeds = self.visual_projection(pooled_output)
        
        
        return CLIPVisionModelOutput(
            image_embeds = image_embeds,
            last_hidden_state = vision_outputs.last_hidden_state,
            hidden_states = vision_outputs.hidden_states,
            attentions = vision_outputs.attentions)













