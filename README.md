# 🏰 Zork GPT Playground

A modern, AI-enhanced version of the classic Zork 1 text adventure. 
Play with **Ollama** (local), **OpenAI**, **Gemini**, **Claude**, or **Groq**.
Generate images of your adventure using **DALL-E 3** or **Stable Diffusion**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/GPT-zork1/blob/main/zork_colab.ipynb)

## Features
- **Multi-Provider Support**: Switch between LLMs on the fly.
- **Image Generation**: See the world of Zork come to life.
- **Customizable**: Edit the "Dungeon Master" system prompt.
- **Local First**: Designed to run with Ollama for privacy.

## How to Run Locally

1. **Install Ollama**: [Download here](https://ollama.com/) and run `ollama serve`.
2. **Pull a Model**: `ollama pull llama3`
3. **Run the App**:
   
   **Option A: Quick Start (Linux/Mac)**
   ```bash
   ./run.sh
   ```

   **Option B: Manual Setup**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run
   python app.py
   ```

4. Open your browser at `http://localhost:7860`.

## How to Run on Colab
Click the "Open in Colab" badge above. It will install everything and give you a public link to play.

## Deployment (Hugging Face Spaces)
1. Create a new Space (SDK: Docker).
2. Upload these files.
3. Enjoy!
