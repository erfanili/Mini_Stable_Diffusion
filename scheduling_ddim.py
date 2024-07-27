from typing import Dict, List, Optional, Union
import numpy as np 
import torch
from configuration_utils import register_to_config

#130
@register_to_config
class DDIMScheduler():
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        set_alpha_to_one: bool = True,
    ):
        
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype = torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype = torch.float32)
        
        
        
        #219
        
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim = 0)
        
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        
        self.init_noise_sigma = 1.0
        
        self.num_inference_steps = None
        
        self.timesteps = torch.from_numpy(np.arange(0,num_train_timesteps)[::-1].copy().astype(np.int64))
    
    #252
    def _get_variance(self,timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 -alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1- alpha_prod_t / alpha_prod_t_prev)
        
        return variance
        
    
    #341
    def step(
        self,
        model_output: torch.Tensor,
        timestep:int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator = None,
        variance_noise: Optional[torch.Tensor] = None
    ):
        
        #401
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        
        
        
        #436
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance **(0.5)
        
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t **(0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t **(0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t ** 0.5) * model_output 
            pred_epsilon = (alpha_prod_t **0.5) * model_output + (beta_prod_t ** 0.5) * sample 
            
            
            
            
        #444
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) **(0.5 * pred_epsilon)
        
        prev_sample = alpha_prod_t_prev **(0.5) * pred_original_sample + pred_sample_direction 
        
        
        return (prev_sample,)