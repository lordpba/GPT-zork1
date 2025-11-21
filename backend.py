import providers
import image_gen

class GameEngine:
    def __init__(self):
        self.providers = {
            "Ollama": providers.OllamaProvider(),
            "OpenAI": providers.OpenAIProvider(),
            "Gemini": providers.GeminiProvider(),
            "Claude": providers.AnthropicProvider(),
            "Groq": providers.GroqProvider()
        }
        self.image_providers = {
            "OpenAI": image_gen.OpenAIImageProvider(),
            "Gemini": image_gen.GeminiImageProvider()
        }
        
        self.default_system_prompt = """You are the Dungeon Master for a text adventure game based on Zork 1. 
Your goal is to simulate the game world, describe locations, track inventory, and handle user commands.
Be descriptive, atmospheric, and immersive. 
If the user asks to do something impossible, explain why.
Keep the tone consistent with the original Zork: mysterious, slightly witty, and perilous.
Start by describing the opening scene: "West of House".
Do not break character.
"""

    def get_provider(self, name):
        return self.providers.get(name)

    def get_image_provider(self, name):
        return self.image_providers.get(name)

    def chat(self, message, history, text_provider_name, image_provider_name, 
             text_model, text_api_key, image_api_key, system_prompt):
        
        if not system_prompt:
            system_prompt = self.default_system_prompt

        # 1. Generate Text
        provider = self.get_provider(text_provider_name)
        if not provider:
            return "Error: Invalid Text Provider", None
        
        response_text = provider.generate(
            system_prompt, 
            history, 
            message, 
            model_name=text_model, 
            api_key=text_api_key
        )

        # 2. Generate Image (Optional)
        image_url = None
        if image_provider_name and image_provider_name != "None":
            # Logic for "Auto" could go here, but we'll stick to explicit selection for now 
            # or map "Auto" in the UI to a specific provider.
            img_prov = self.get_image_provider(image_provider_name)
            if img_prov:
                # We use the response text to generate the image prompt, 
                # or the user message? Usually the scene description (response) is better.
                # Let's use a summarized version of the response for the image prompt.
                image_prompt = response_text[:300] # Simple truncation for now
                image_url = img_prov.generate_image(image_prompt, api_key=image_api_key)

        return response_text, image_url
