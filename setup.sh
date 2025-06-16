#!/bin/bash
# setup.sh - Setup script for the Chainlit Knowledge Graph Chatbot

echo "🚀 Setting up Chainlit Knowledge Graph Chatbot..."

# Create project directory structure
echo "📁 Creating project structure..."
mkdir -p .chainlit/public
mkdir -p logs

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install chainlit>=1.0.0 openai>=1.0.0 networkx>=3.0 numpy>=1.24.0 python-dotenv>=1.0.0

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔐 Creating .env file..."
    cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Customize the model
OPENAI_MODEL=gpt-4o

# Optional: Enable debug mode
DEBUG=false
EOF
    echo "⚠️  Please edit .env file and add your OpenAI API key!"
fi

# Create custom CSS for better Chinese font support
echo "🎨 Creating custom CSS..."
cat > .chainlit/public/style.css << EOF
/* Custom styling for Chinese text support */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

body {
    font-family: 'Noto Sans TC', 'Microsoft YaHei', sans-serif !important;
}

.MuiTypography-root {
    font-family: 'Noto Sans TC', 'Microsoft YaHei', sans-serif !important;
}

/* Enhance message display */
.cl-message {
    line-height: 1.6 !important;
}

/* Better button styling */
.cl-action-button {
    margin: 4px !important;
    border-radius: 8px !important;
}

/* Enhance step display */
.cl-step {
    margin: 8px 0 !important;
    padding: 12px !important;
    border-radius: 8px !important;
    background-color: #f8f9fa !important;
}

/* Knowledge graph entity styling */
.kg-entity {
    background-color: #e3f2fd;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
    margin: 0 2px;
}

/* Relationship text styling */
.kg-relationship {
    color: #1976d2;
    font-weight: 500;
}
EOF

# Check if knowledge graph file exists
if [ ! -f "knowledge_graph.json" ]; then
    echo "⚠️  knowledge_graph.json not found!"
    echo "   Please ensure you have the knowledge graph file in the current directory."
    echo "   You can generate it using UChatbot_v5.py"
fi

# Create launch script
echo "🚀 Creating launch script..."
cat > run_chatbot.sh << EOF
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
EOF

chmod +x run_chatbot.sh

# Create requirements.txt
echo "📋 Creating requirements.txt..."
cat > requirements.txt << EOF
chainlit>=1.0.0
openai>=1.0.0
networkx>=3.0
numpy>=1.24.0
python-dotenv>=1.0.0
asyncio
logging
re
datetime
typing
EOF

echo ""
echo "✅ Setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Ensure knowledge_graph.json is in the current directory"
echo "3. Run: ./run_chatbot.sh"
echo ""
echo "🔧 Optional customizations:"
echo "• Edit .chainlit/config.toml for UI settings"
echo "• Modify .chainlit/public/style.css for styling"
echo "• Check logs/ directory for error logs"
echo ""

