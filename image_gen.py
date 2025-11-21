import os
from abc import ABC, abstractmethod
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

class ImageProvider(ABC):
    @abstractmethod
    def generate_image(self, prompt, api_key=None):
        pass

class OpenAIImageProvider(ImageProvider):
    def generate_image(self, prompt, api_key=None):
        if not api_key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=f"Retro text adventure game art, fantasy style. {prompt}",
                size="1024x1024",
                quality="standard",
                n=1,
            )
            return response.data[0].url
        except Exception as e:
            print(f"OpenAI Image Error: {e}")
            return None

class GeminiImageProvider(ImageProvider):
    def generate_image(self, prompt, api_key=None):
        # Note: Gemini Image generation via API might vary by region/version.
        # This is a placeholder for the standard implementation once widely available via the python SDK cleanly.
        # Currently, it's often safer to return None or use a specific beta endpoint.
        # For now, we will implement a stub or try the standard call if available.
        return None 

class LocalStableDiffusionProvider(ImageProvider):
    def __init__(self):
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            try:
                import torch
                from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
                # Using a fast model for local use
                base = "stabilityai/stable-diffusion-xl-base-1.0"
                repo = "ByteDance/SDXL-Lightning"
                ckpt = "sdxl_lightning_4step_unet.safetensors"
                
                # Load UNet
                unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
                unet.load_state_dict(torch.load(hf_hub_download(repo, ckpt), map_location="cuda"))
                
                self.pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
                self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
            except Exception as e:
                print(f"Failed to load local SD: {e}")

    def generate_image(self, prompt, api_key=None):
        # This requires GPU and heavy dependencies. 
        # We will catch import errors gracefully.
        try:
            if self.pipe is None:
                # Lazy loading or just fail if not set up
                return None
            
            image = self.pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
            return image
        except Exception:
            return None
