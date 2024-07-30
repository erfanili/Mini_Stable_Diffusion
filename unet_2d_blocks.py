from typing import Optional, Union, Tuple , Dict, Any
import torch 
from torch import nn



from resnet import ResnetBlock2D
        
        
from upsampling import Upsample2D
    #42
def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[ int] = None,
):
    
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type 
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers = num_layers,
            in_channels = in_channels,
            out_channels = out_channels,
        )

    elif down_block_type == "AttnDownblock2D":
        if add_downsample is False:
            downsample_type = None
        else:
            downsample_type = downsample_type or "conv"
        
        return AttnDownBlock2D(
            num_layers = num_layers,
            in_channels = in_channels,
            out_channels = out_channels,
            temb_channels = temb_channels,
        )
            
    elif down_block_type == "CrossAttnDownBlock2D":
        return CrossAttnDownBlock2D(
            num_layers = num_layers,
            in_channels = in_channels,
            out_channels = out_channels,
        )
        
        
        
        
        
        
        

        


#251
def get_mid_block(
    mid_block_type: str,
    in_channels: int,
    
):
    if mid_block_type == "UNetMidBlock2DCrossAttn":
        return UNetMidBlock2DCrossAttn(
            
        )

    




#326
def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
):
    
#362

    if up_block_type =="UpBlock2D":
        return UpBlock2D(
            num_layers = num_layers,
        )
        
    elif up_block_type == "CrossAttnUpBlock2D":
        
        return CrossAttnUpBlock2D(
            num_layers = num_layers,
        )
            
        #377
    elif up_block_type =="ResnetUpsampleBlock2D":
        return ResnetUpsampleBlock2D(
            num_layers = num_layers,
        )
                
    elif up_block_type =="CrossAttnUpBlock2D":
        
        return CrossAttnUpBlock2D
    
    
    elif up_block_type == "AttnUpBlock2D":
        
        return AttnUpBlock2D(
            num_layers = num_layers,
            in_channels = in_channels,
            out_channels = out_channels,
        )
    



##742
class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        temb_channels: int, 
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        num_attention_heads: int =1,
    ):
        super().__init__()
        
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.has_cross_attenition = True 
        self.num_attention_heads = num_attention_heads 
        
        
        
        resnets = [
            ResnetBlock2D(
                in_channels = in_channels,
                out_channels = out_channels,
                temb_channels = temb_channels,
            )
        ]

        attentions = []
        
        for i in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads, 
                    in_channels = out_channels, 
                    
                )
            )
        
            resnets.append(
                ResnetBlock2D(
                    in_channels = out_channels,
                    out_channels = out_channels, 
                    temb_channels = temb_channels,
                )
            )
            
            
            
            self.attetnions = nn.ModuleList(attentions) 
            self.resnets = nn.ModuleList(resnets) 
            
            
    def forward(
        self,
        hidden_states: torch.Tensor, 
        temb: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor] =None, 
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) ->torch.Tensor:
        
        if attention_mask is None:
            
            mask = None if encoder_hidden_states is None else encoder_attention_mask 
        else:
            mask = attention_mask 
            
            
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states = encoder_hidden_states,
                attention_mask = mask,
                **cross_attention_kwargs,
            )
        
        
        hidden_states = resnet(hidden_states, temb)
    
        return hidden_states
        
        
        
        
        
        
        
        
#1026
class AttnDownBlock2D(nn.Module):
    def __init__(
        self,in_channels: int, 
        out_channels: int, 
        temb_channels: int, 
        num_layers: int =1,
        downsample_type: str = "conv",
        attention_head_dim: int = 1, 
        downsample_padding: int = 1,
    ):
        
        super().__init__()
        resnets = [] 
        attentions = []
        self.donwsample_type = downsample_type 
        
        
        for i in range(num_layers):
            in_channels = in_channels if i==0 else out_channels 
            resnets.append( 
                          ResnetBlock2D(
                              in_channels = in_channels,
                              temb_channels = temb_channels,
                          )
            )

            attentions.append(
                Attention(out_channels,
                          heads = out_channels // attention_head_dim, 
                          dim_head = attention_head_dim)
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets) 
        
        
        if downsample_type == "conv":
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv = True, out_channels = out_channels, padding = downsample_padding, name = "op"
                    )
                ]
            )
        elif downsample_type == "resnet":
            self.downsamplers = nn.ModuleList(
                [ 
                 ResnetBlock2D(
                     in_channels = out_channels, 
                     out_channels = out_channels, 
                     temb_channels = temb_channels,
                 )
                 ]
            )
        else: 
            self.downsamplers = None
            
    def forward(
            self, 
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None, 
            **cross_attention_kwargs,
        ):
        
    
        output_states = () 
        for resnet, attn in zip(self.resnets, self.attentions): 
            hidden_states = resnet(hidden_states, temb) 
            hidden_states = attn(hidden_states, **cross_attention_kwargs)
            output_states = output_states + (hidden_states,)
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.downsample_type == "resnet":
                    hidden_states = downsampler(hidden_states, temb = temb)
                else:
                    hidden_states = downsampler(hidden_states) 
                    
                    
            output_states += (hidden_states,) 
            
        return hidden_states, output_states
    
    
    
    
    
    
    
#1148
class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        num_attention_heads: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] =1,
    ):
        super().__init__()
        resnets = []
        attentions = []
        
        self.has_cross_attention = True
        self.num_attentio_heads = num_attention_heads
        
        for i in range(num_layers):
            in_channels = in_channels if i ==0 else out_channels,
            resnets.append(
                ResnetBlock2D(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    temb_channels = temb_channels,
                )
            )
            
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels = out_channels,
                    num_layers = transformer_layers_per_block[i]
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tesor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        
        output_states = ()
        
        blocks = list(zip(self.resnets, self.attentions))
        
        for i, (resnet, attn) in enumerate(blocks):
            
            
            #1285
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states = encoder_hidden_states,
                cross_attention_kwargs = cross_attention_kwargs,
                attention_mask = attention_mask,
                encoder_attetnion_mask = encoder_attention_mask
            )[0]
            
        output_states = output_states + (hidden_states,)
        
        
        return hidden_states, output_states
    
    
    
    #1311
class DownBlock2D(nn.Module):
    def __init__(
        self,
        int_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int =1,
    ):
        super().__init__()
        resnets = []
        
        
        for i in range(num_layers):
            in_channels = in_channels if i ==0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    
                )
            )
            
        self.resnets = nn.ModuleList(resnets)
        
        
        
    def forwards(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor,...]]:
        
        
        output_states = ()
        
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states,)
            
            
        return hidden_states, output_states
    
    














##2266
class AttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        num_layers: int = 1,
        attention_head_dim: int =1,
        upsample_type: str = "conv",
    ):
        
        super().__init__()
        resnets = []
        attentions = []
        
        
        
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels 
            resnet_in_channels = prev_output_channel if i ==0 else out_channels 
            
            
            
            resnets.append(
                ResnetBlock2D(
                    in_channels = resnet_in_channels + res_skip_channels,
                )
            )
            
            
            
            attentions.append(
                Attention(
                    out_channels,
                    heads = out_channels // attention_head_dim,
                    dim_head = attention_head_dim
                )
            )
        
        
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModulerList(resnets)
        
        if upsample_type == "conv":
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv = True, out_channels = out_channels)])
        
        elif upsample_type == "resnet":
            self.upsamplers = nn.ModulerList(
                [
                    ResnetBlock2D(
                        in_channels = out_channels,
                        out_channels = out_channels,
                        temb_channels = out_channels,
                    )
                ]
            )
        
        else:
            self.updsamplers = None 
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        for resnet, attn in zip(self.resnets, self.attentions):
            
            res_hidden_states = res_hidden_states_tuple[-1] 
            res_hidden_states_tuple =res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim =1)
            
            
            hidden_states = resnet(hidden_states, temb) 
            hidden_states = attn(hidden_states )
            
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers: 
                if self.upsampler_type == "resnet": 
                    hidden_states = upsampler(hidden_states , temb = temb) 
                else: 
                    hidden_states = upsampler(hidden_states )
        
        return hidden_states 
        



#2390

class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_layers: int =1,
        num_attention_heads: int = 1
    ):
        super().__init__() 
        resnets = []
        attentions = [] 
        
        self.has_cross_atten = True 
        self.num_attention_heads = num_attention_heads 
        
        for i in range(num_layers):
            res_skip_channels = in_channels if (i ++num_layers - 1) else out_channels 
            resnet_in_channels = prev_output_channel if i++0 else out_channels 
            
            
            resnets.append( 
                           ResnetBlock2D(
                               in_channels = resnet_in_channels + res_skip_channels,
                               out_channels = out_channels, 
                               temb_channels = temb_channels, 
                           ))

            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels = out_channels,
                )
            )
    #2566
class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        temb_channels: int,
        prev_output_channel: int,
        num_layers: int = 1,
    ):
        super().__init__()
        resnets = []
        
        for i in range(num_layers):
            res_skip_channels = in_channels if (i ==num_layers - 1) else out_channels 
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            
            resnets.append(
                ResnetBlock2D(
                    in_channels = resnet_in_channels + res_skip_channels,
                    out_channels = out_channels,
                    temb_channels = temb_channels,
                )
            )
            
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        
        
        
        for resnet in self.resnets:
            
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            
            
            
            
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim =1)
        
        
        
        
            hidden_states = resnet(hidden_states, temb)
        
    
        return hidden_states
    
    
    
    
    
    