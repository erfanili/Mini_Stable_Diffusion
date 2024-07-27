import numpy as np 
from typing import Union, List, Optional, Tuple
import PIL.Image 
from PIL import Image
import torch 
import torch.nn.functional as F


class VaeImageProcessor():
    def __init__(
        self,
    ):
        super().__init__()
        
    def numpy_to_pil(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images] 
        images = np.stack(images, axis = 0)
        
        return images 
    
    @staticmethod
    def rgblike_to_depthmap(image:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        
        return image[:,:,1] * 2**8 + image[:,:,2]
    
    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        
        return (images / 2 + 0.5).clamp(0,1)
    
    def pt_to_numpy(images: torch.Tensor)-> np.ndarray:
        
        images = images.cpu().permute(0,2,3,1).float().numpy()
        
        return images
    
    
    def numpy_to_depth(self,images:np.ndarray) -> List[PIL.Image.Image]:
        
        if images.ndim == 3:
            images = images[None,...]
        images_depth = images[:,:,:,3:]
        if images.shape[-1] ==6:
            images_depth = (images_depth * 255).round().astype("uint8")
            pil_images = [
                Image.fromarray(self.rgblike_to_depthmap(image_depth), mode = "I;16") for image_depth in images_depth
            ]
            
        elif images.shape[-1] ==4:
            images_depth = (images_depth * 65535.0).astype(np.uint16)
            pil_images = [Image.fromarray(image_depth, mode = "I;16") for image_depth in images_depth]
            
        return pil_images
    
    
    
    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        
        
        if output_type == "latent":
            return image 
        
        if do_denormalize is None:
            do_denormalize = [self.config.do_denormalize] * image.shape[0]
            
        image =  torch.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )
        
        
        if output_type == "pt":
            return image 
        image = self.pt_to_numpy(image)
        
        if output_type == "np":
            return image 
        
        if output_type == "pil":
            return self.numpy_to_pil(image)




        if output_type == "pil":
            return self.numpy_to_pil(image), self.numpy_to_depth(image)    
        
        