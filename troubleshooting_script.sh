#!/bin/bash
# troubleshoot.sh - Fix OpenAI compatibility issues

echo "🔧 Troubleshooting OpenAI compatibility issues..."
echo ""

# Check Python version
echo "1️⃣ Checking Python version..."
python_version=$(python --version 2>&1)
echo "   Python version: $python_version"

# Check if we're in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "   ✅ Virtual environment: $VIRTUAL_ENV"
else
    echo "   ⚠️  Not in virtual environment (recommended to use one)"
fi
echo ""

# Check OpenAI library
echo "2️⃣ Checking OpenAI library..."
if python -c "import openai; print('OpenAI version:', openai.__version__)" 2>/dev/null; then
    openai_version=$(python -c "import openai; print(openai.__version__)")
    echo "   Current OpenAI version: $openai_version"
    
    # Check if it's the old version causing issues
    if [[ "$openai_version" < "1.0.0" ]]; then
        echo "   ⚠️  Old OpenAI version detected. Upgrading..."
        pip install --upgrade "openai>=1.0.0"
    else
        echo "   ✅ OpenAI version is compatible"
    fi
else
    echo "   ❌ OpenAI not installed. Installing..."
    pip install "openai>=1.0.0"
fi
echo ""

# Check API key
echo "3️⃣ Checking OpenAI API key..."
if [[ -n "$OPENAI_API_KEY" ]]; then
    if [[ "$OPENAI_API_KEY" == sk-* ]]; then
        echo "   ✅ API key format looks correct"
    else
        echo "   ⚠️  API key format may be incorrect (should start with 'sk-')"
    fi
else
    echo "   ❌ OPENAI_API_KEY environment variable not set"
    echo "   Please set it with: export OPENAI_API_KEY='sk-your-key-here'"
fi
echo ""

# Check knowledge graph file
echo "4️⃣ Checking knowledge graph file..."
if [[ -f "knowledge_graph.json" ]]; then
    file_size=$(wc -c < knowledge_graph.json)
    echo "   ✅ knowledge_graph.json exists (${file_size} bytes)"
    
    # Validate JSON format
    if python -c "import json; json.load(open('knowledge_graph.json'))" 2>/dev/null; then
        echo "   ✅ JSON format is valid"
    else
        echo "   ❌ JSON format is invalid"
    fi
else
    echo "   ❌ knowledge_graph.json not found"
    echo "   Please ensure the file exists in the current directory"
fi
echo ""

# Check required dependencies
echo "5️⃣ Checking other dependencies..."
dependencies=("chainlit" "networkx" "numpy")

for dep in "${dependencies[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        version=$(python -c "import $dep; print($dep.__version__)" 2>/dev/null || echo "unknown")
        echo "   ✅ $dep ($version)"
    else
        echo "   ❌ $dep not installed"
        echo "      Installing $dep..."
        pip install "$dep"
    fi
done
echo ""

# Test OpenAI connection
echo "6️⃣ Testing OpenAI connection..."
if [[ -n "$OPENAI_API_KEY" ]]; then
    cat > test_openai.py << 'EOF'
import os
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test a simple completion
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("✅ OpenAI API connection successful")
    print(f"   Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ OpenAI API connection failed: {e}")
    
    # Try legacy API
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("✅ Legacy OpenAI API works")
    except Exception as e2:
        print(f"❌ Legacy API also failed: {e2}")
EOF

    python test_openai.py
    rm test_openai.py
else
    echo "   ⚠️  Skipping connection test (no API key)"
fi
echo ""

# Create fixed requirements.txt
echo "7️⃣ Creating fixed requirements.txt..."
cat > requirements_fixed.txt << EOF
# Fixed requirements for Chainlit Knowledge Graph Chatbot
chainlit>=1.0.0
openai>=1.0.0,<2.0.0
networkx>=3.0
numpy>=1.24.0
python-dotenv>=1.0.0

# Optional for better performance
aiofiles>=0.8.0
asyncio-throttle>=1.0.0
EOF

echo "   ✅ Created requirements_fixed.txt"
echo ""

# Installation commands
echo "8️⃣ Recommended installation commands:"
echo ""
echo "   # Clean install (recommended):"
echo "   pip uninstall openai -y"
echo "   pip install -r requirements_fixed.txt"
echo ""
echo "   # Or force upgrade:"
echo "   pip install --upgrade --force-reinstall openai>=1.0.0"
echo ""

# Create a simple test script
echo "9️⃣ Creating test script..."
cat > test_chatbot.py << 'EOF'
#!/usr/bin/env python
"""Test script for the knowledge graph chatbot"""

import os
import sys

def test_imports():
    """Test all required imports"""
    try:
        import chainlit as cl
        print("✅ Chainlit imported successfully")
    except ImportError as e:
        print(f"❌ Chainlit import failed: {e}")
        return False
    
    try:
        from openai import OpenAI
        print("✅ OpenAI v1+ imported successfully")
        return True
    except ImportError:
        try:
            import openai
            print("✅ OpenAI legacy imported successfully")
            return True
        except ImportError as e:
            print(f"❌ OpenAI import failed: {e}")
            return False

def test_knowledge_graph():
    """Test knowledge graph loading"""
    try:
        import json
        import networkx as nx
        from networkx.readwrite import json_graph
        
        with open('knowledge_graph.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        graph = json_graph.node_link_graph(data)
        print(f"✅ Knowledge graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return True
    except FileNotFoundError:
        print("❌ knowledge_graph.json not found")
        return False
    except Exception as e:
        print(f"❌ Knowledge graph loading failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        print("✅ OpenAI API connection successful")
        return True
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        
        # Try legacy API
        try:
            import openai
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            print("✅ Legacy OpenAI API works")
            return True
        except Exception as e2:
            print(f"❌ Legacy API also failed: {e2}")
            return False

def main():
    """Run all tests"""
    print("🧪 Testing Chainlit Knowledge Graph Chatbot Setup")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Knowledge Graph", test_knowledge_graph),
        ("OpenAI Connection", test_openai_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! Your chatbot should work correctly.")
        print("   Run: chainlit run fixed_chainlit_chatbot.py -w")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("   Run the troubleshoot.sh script for help.")

if __name__ == "__main__":
    main()
EOF

echo "   ✅ Created test_chatbot.py"
echo ""

# Final recommendations
echo "🎯 **SOLUTION SUMMARY**"
echo "========================"
echo ""
echo "The 'proxies' error usually indicates an OpenAI library version conflict."
echo "Here's how to fix it:"
echo ""
echo "**Option 1: Quick Fix**"
echo "   pip uninstall openai -y"
echo "   pip install 'openai>=1.0.0,<2.0.0'"
echo ""
echo "**Option 2: Use the Fixed Chatbot**"
echo "   Use 'fixed_chainlit_chatbot.py' instead of the enhanced version"
echo "   It handles both old and new OpenAI API versions automatically"
echo ""
echo "**Option 3: Clean Environment**"
echo "   python -m venv venv_new"
echo "   source venv_new/bin/activate  # On Windows: venv_new\\Scripts\\activate"
echo "   pip install -r requirements_fixed.txt"
echo ""
echo "**Test Your Setup:**"
echo "   python test_chatbot.py"
echo ""
echo "**Run the Fixed Chatbot:**"
echo "   export OPENAI_API_KEY='sk-your-key-here'"
echo "   chainlit run fixed_chainlit_chatbot.py -w"
echo ""
echo "🔗 **Helpful Links:**"
echo "   • OpenAI Python Library: https://github.com/openai/openai-python"
echo "   • Chainlit Docs: https://docs.chainlit.io/"
echo ""

# Check if we can provide specific fix
echo "🚀 **IMMEDIATE ACTION:**"
if command -v pip &> /dev/null; then
    echo "   Running automatic fix..."
    pip uninstall openai -y >/dev/null 2>&1
    pip install "openai>=1.0.0,<2.0.0" >/dev/null 2>&1
    echo "   ✅ OpenAI library updated"
    echo ""
    echo "   Now run: python test_chatbot.py"
else
    echo "   Please run the pip commands above manually"
fi

echo ""
echo "✨ Troubleshooting completed!"
