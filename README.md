---
title: Print&Gift Chatbot
emoji: 🎁
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Print&Gift Chatbot

A PHP-based chatbot interface for Print&Gift Singapore, powered by local Ollama LLM.

## Features

- Interactive chat interface
- Product recommendations
- Image upload support
- Responsive design
- Integration with local Ollama models

## Setup

1. Set the `OLLAMA_URL` environment variable to your public Ollama endpoint
2. Optionally set `OLLAMA_MODEL` (default: llama2)
3. The application will be available at the space URL

## Environment Variables

- `OLLAMA_URL`: Your public Ollama API endpoint (required)
- `OLLAMA_MODEL`: Model to use (optional, default: llama2)
- `DEBUG_MODE`: Enable debug logging (optional, default: false)