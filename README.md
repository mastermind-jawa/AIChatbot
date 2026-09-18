# AI Chatbot

An interactive AI Chatbot web application built with Flask, integrating Groq and Google Gemini APIs with document analysis and conversational capabilities.

## Features

- 💬 Interactive chat interface with real-time AI responses
- ⚡ Powered by Groq and Google Gemini models
- 📄 Document and PDF processing support
- 🎨 Modern web UI

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Groq API Key and/or Google Gemini API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mastermind-jawa/AI-Chatbot.git
   cd AI-Chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env
     ```
   - Add your API keys to `.env`:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://localhost:5000`.
