import gradio as gr
import backend

engine = backend.GameEngine()

def get_models_for_provider(provider_name, api_key):
    provider = engine.get_provider(provider_name)
    if provider:
        return gr.update(choices=provider.get_models(api_key), value=provider.get_models(api_key)[0] if provider.get_models(api_key) else None)
    return gr.update(choices=[], value=None)

def game_turn(message, history, text_prov, img_prov, txt_model, txt_key, img_key, sys_prompt):
    response, image = engine.chat(
        message, 
        history, 
        text_prov, 
        img_prov, 
        txt_model, 
        txt_key, 
        img_key, 
        sys_prompt
    )
    return response

def game_turn_with_image(message, history, text_prov, img_prov, txt_model, txt_key, img_key, sys_prompt):
    # Gradio ChatInterface expects just text return usually, but we want to display an image too.
    # We might need a custom Chatbot or just append the image to the chat.
    # For simplicity in this "Playground", let's return the text to the chat 
    # and update a separate Image component.
    
    response, image_data = engine.chat(
        message, 
        history, 
        text_prov, 
        img_prov, 
        txt_model, 
        txt_key, 
        img_key, 
        sys_prompt
    )
    return response, image_data

with gr.Blocks(title="Zork GPT Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏰 Zork GPT Playground")
    gr.Markdown("Play Zork enhanced by LLMs. Choose your provider and model below.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            # Text Provider Settings
            text_provider = gr.Dropdown(
                choices=["Ollama", "OpenAI", "Gemini", "Claude", "Groq"], 
                value="Ollama", 
                label="Text Provider"
            )
            text_api_key = gr.Textbox(label="Text API Key", type="password", placeholder="Leave empty for Ollama")
            text_model = gr.Dropdown(label="Model", allow_custom_value=True)
            
            # Refresh models button
            refresh_btn = gr.Button("🔄 Refresh Models")
            
            gr.Markdown("---")
            
            # Image Provider Settings
            image_provider = gr.Dropdown(
                choices=["None", "OpenAI", "Gemini", "Local (SD)"], 
                value="None", 
                label="Image Provider"
            )
            image_api_key = gr.Textbox(label="Image API Key", type="password")
            
            gr.Markdown("---")
            system_prompt = gr.Textbox(
                label="System Prompt", 
                value=engine.default_system_prompt,
                lines=5
            )

        with gr.Column(scale=3):
            # Image Display
            scene_image = gr.Image(label="Current Scene", interactive=False, height=300)
            
            # Chat Interface
            chatbot = gr.Chatbot(height=500, type="messages")
            msg = gr.Textbox(label="Your Command", placeholder="open mailbox, go north...")
            clear = gr.Button("Clear")

            # State
            history = gr.State([])

            def user(user_message, history):
                return "", history + [{"role": "user", "content": user_message}]

            def bot(history, text_prov, img_prov, txt_model, txt_key, img_key, sys_prompt):
                user_message = history[-1]["content"]
                # Pass the history excluding the last message which is the current user input
                # Actually engine.chat expects history as list of [user, bot] tuples or similar.
                # Let's adapt.
                
                # Convert new 'messages' format to old list of lists format for the engine if needed
                # Or update engine to handle list of dicts. 
                # The engine currently expects: list of [user_text, bot_text]
                
                old_history_format = []
                for i in range(0, len(history)-1, 2):
                    if i+1 < len(history):
                        old_history_format.append([history[i]["content"], history[i+1]["content"]])
                
                response_text, image_url = engine.chat(
                    user_message, 
                    old_history_format, 
                    text_prov, 
                    img_prov, 
                    txt_model, 
                    txt_key, 
                    img_key, 
                    sys_prompt
                )
                
                history.append({"role": "assistant", "content": response_text})
                return history, image_url

            msg.submit(user, [msg, history], [msg, history], queue=False).then(
                bot, 
                [history, text_provider, image_provider, text_model, text_api_key, image_api_key, system_prompt], 
                [chatbot, scene_image]
            )
            clear.click(lambda: None, None, chatbot, queue=False)

    # Event Listeners
    text_provider.change(get_models_for_provider, [text_provider, text_api_key], [text_model])
    refresh_btn.click(get_models_for_provider, [text_provider, text_api_key], [text_model])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
