import os
from abc import ABC, abstractmethod


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
