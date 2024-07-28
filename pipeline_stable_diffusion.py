












import inspect
from typing import Optional, Union, List

import torch







from autoencoder_kl import AutoencoderKL
from modeling_clip import CLIPTextModel, CLIPVisionModelWithProjection
from torch_utils import randn_tensor
from image_processor import VaeImageProcessor
from tokenization_clip import CLIPTokenizer
from unet_2d_condition import UNet2DConditionModel    
from scheduling_ddim import DDIMScheduler
from clip_image_processor import CLIPImageProcessor


























def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale = 0.0):
    
    
    
    
    std_text = noise_pred_text.std(dim = list(range(1, noise_pred_text.ndim)), keepdim = True)
    std_cfg = noise_cfg.std(dim = list(range(1,noise_cfg.ndim)), keepdim = True)
    
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg 
    return noise_cfg 


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of 'timesteps' or 'sigmas' can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s 'set_timesteps' does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps = timesteps, device = device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class_}'s 'timesteps' does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas = sigmas, device = device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device = device, ** kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps 


class StableDiffusionPipeline():
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def __init__(
        self,
        vae:AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,    
    ):
        super().__init__()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.register_modules(
            vae = vae,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            unet = unet,
            scheduler = scheduler,
            
            feature_extractor = feature_extractor,
            image_encoder = image_encoder,
        )
        self.vae_scale_factor = 2 **(len(self.vae.config.bloc_out_channels)-1)
        self.image_processor = VaeImageProcessor(vae_scale_factor = self.vae_scale_factor)
        self.register_to_config(requires_safety_checker = requires_safety_checker)
        
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt = None,
        prompt_embeds: Optional[torch.Tensor] =None,
        negative_prompt_embeds: Optional[torch.Tnesor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        
        
        
        prompt_embeds_tuple = self.encode_prompt(
            prompt = prompt,
            device = device,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance = do_classifier_free_guidance,
            negative_prompt = negative_prompt,
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            lora_scale = lora_scale,
            **kwargs,
        )
        

        prompt_embeds = torch.cat(prompt_embeds_tuple[1],prompt_embeds_tuple[0])
        
        return prompt_embeds
        
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        if prompt_embeds is None:
            
            
            
            
            text_inputs = self.tokenizer(
                prompt,
                padding = "max_length",
                max_length = self.tokenizer.model_max_length,
                truncation = True,
                return_tensors = "pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding = "longest", return_tensors = "pt").input_ids
            
            
            
            
            
            
            
            
            
            
            
            if hasattr(self.text_encoder.config, "use_attention_maks") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            
            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask = attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask = attention_mask, output_hidden_states = True
                )
                


                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                
                
                
                
                prompt_embeds = self.text_encoder.text_model.funal_layer_norm(prompt_embeds)
        
        
        
        
        
        
        
            
        prompt_embeds = prompt_embeds.to(device = device)
        
        bs_embed, seq_len, _ = prompt_embeds.shape
        
        prompt_embeds= prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size 





            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            
            
            
            
            
            
            else:
                uncond_tokens = negative_prompt
                
                
            
            
            
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding = "max_length",
                max_length = max_length,
                truncation = True,
                return_tensors = "pt",
            )
        
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
                
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask = attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            
        if do_classifier_free_guidance:
            
            seq_len = negative_prompt_embeds.shape[1]
            
            negative_prompt_embds = negative_prompt_embeds.to(device = device)
            
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len , -1)
            
        
        
        
        
        
        return prompt_embeds, negative_prompt_embeds
    
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states = None):
        
        
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors = "pt").pixel_values
            
        image = image.to(device = device)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states = True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim = 0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states = True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim = 0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim = 0)
            uncond_image_embeds = torch.zeros_like(image_embeds)
            
            return image_embeds, uncond_image_embeds
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def decode_latents(self, latents):
        
        
        
        latents = 1/ self.vae.config.scaling_factor * latents 
        image = self.vae.decode(latents, return_dict = False)[0]
        image = (image / 2 + 0.5).clamp(0,1)
        
        image = image.cpu().permute(0,2,3,1).float().numpy()
        return image
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def prepare_latents(self,batch_size, num_channels_latents, height, width, dtype, device, generator, latents= None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        
        
        
        
        
        
        if latents is None:
            latents = randn_tensor(shape, generator = generator, device = device, dtype = dtype)
        else:
            latents = latents.ro(device)
            return latents
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[ torch.Generator]]] = None,
        latents: Optional[ torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negatove_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self._guidance_scale = guidance_scale
        
        
        
        
        
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and  isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
            device = self._executon_device
            
            
            
            
            
            
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompr_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
        )
        
        
        
        
        
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
            
            
            
            
            
            
            
            
            
            
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
                
                
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )        
        
        
        
        
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total = num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states = prompt_embeds,
                )[0]
                
                
                
                
                
                
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        
        
        
        
        
                latents = self.scheduler.step(noise_pred, t , latents)[0]
        
        
        
        
        

            
            
            
            
            
            
            
            
            
            
            
            
            
        image = self.vae.decode(latents / self.vae.config.scaling_factor)[0]
            
            
            
            
            
            
            
            
            
            
            
            
        image = self.image_processor.postprocess(image)
            
            
            
            
            
        return image
        
        
        
        
        
        
        
        
        
        
        
        
        
        