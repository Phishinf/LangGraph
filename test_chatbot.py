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
