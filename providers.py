import os
import requests
import json
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt, history, new_input, model_name=None, api_key=None):
        pass

    @abstractmethod
    def get_models(self, api_key=None):
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def get_models(self, api_key=None):
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                return models
            return ["Error: Could not fetch models"]
        except Exception as e:
            return [f"Error: {str(e)}"]

    def generate(self, system_prompt, history, new_input, model_name="llama3", api_key=None):
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": new_input})

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload)
            if response.status_code == 200:
                return response.json()["message"]["content"]
            return f"Error: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

class OpenAIProvider(LLMProvider):
    def get_models(self, api_key=None):
        return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

    def generate(self, system_prompt, history, new_input, model_name="gpt-4o", api_key=None):
        if not api_key:
            return "Error: API Key required for OpenAI"
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": new_input})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

class GeminiProvider(LLMProvider):
    def get_models(self, api_key=None):
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]

    def generate(self, system_prompt, history, new_input, model_name="gemini-1.5-flash", api_key=None):
        if not api_key:
            return "Error: API Key required for Gemini"
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Gemini handles history differently, but for simplicity we'll construct a prompt or use chat session
        # Using the chat session is better for context
        try:
            model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
            chat = model.start_chat(history=[])
            
            # Replay history to set state (inefficient but stateless for this simple implementation)
            # A better way for Gradio is to keep the chat object in session state, 
            # but here we are stateless per call. 
            # Let's just format it as a single prompt for robustness if the history is long,
            # or reconstruct the history objects.
            
            gemini_history = []
            for h in history:
                gemini_history.append({"role": "user", "parts": [h[0]]})
                gemini_history.append({"role": "model", "parts": [h[1]]})
            
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(new_input)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

class AnthropicProvider(LLMProvider):
    def get_models(self, api_key=None):
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]

    def generate(self, system_prompt, history, new_input, model_name="claude-3-haiku-20240307", api_key=None):
        if not api_key:
            return "Error: API Key required for Anthropic"
        
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        messages = []
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": new_input})

        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

class GroqProvider(LLMProvider):
    def get_models(self, api_key=None):
        return ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]

    def generate(self, system_prompt, history, new_input, model_name="llama3-70b-8192", api_key=None):
        if not api_key:
            return "Error: API Key required for Groq"
        
        from groq import Groq
        client = Groq(api_key=api_key)
        
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": new_input})

        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
