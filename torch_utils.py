import torch 
from typing import Optional, List, Tuple, Union



def randn_tensor(
    shape: Union[tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    layout: Optional["torch.layout"] = None
):
    rand_device = device
    batch_size = shape[0] 
    
    layout = layout or torch.strided 
    device = device or torch.device("cpu")
    
    
    
    latents = torch.randn(shape, generator = generator, device = rand_device, layout = layout).to(device)
    
    return latents