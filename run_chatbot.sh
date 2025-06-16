#!/bin/bash
echo "🎓 Starting Macao Tourism University Knowledge Graph Chatbot..."

# Check if .env exists and has API key
if [ ! -f .env ] || ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "❌ Please set your OpenAI API key in the .env file first!"
    echo "   Edit .env and replace 'your_openai_api_key_here' with your actual API key"
    exit 1
fi

# Check if knowledge graph exists
if [ ! -f "knowledge_graph.json" ]; then
    echo "❌ knowledge_graph.json not found!"
    echo "   Please generate it first using UChatbot_v5.py"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Start the chatbot
echo "✅ Starting Chainlit server..."
echo "🌐 Open http://localhost:8000 in your browser"
chainlit run enhanced_chainlit_chatbot.py -w --host 0.0.0.0 --port 8000
