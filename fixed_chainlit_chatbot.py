import chainlit as cl
import json
import networkx as nx
from networkx.readwrite import json_graph
import os
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import asyncio
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI import with compatibility handling
try:
    from openai import OpenAI
    OPENAI_V1 = True
    logger.info("Using OpenAI v1+ API")
except ImportError:
    try:
        import openai
        OPENAI_V1 = False
        logger.info("Using OpenAI legacy API")
    except ImportError:
        raise ImportError("OpenAI library not found. Please install: pip install openai")

class CompatibleKnowledgeGraphChatbot:
    """Knowledge graph chatbot with OpenAI compatibility handling"""
    
    def __init__(self, kg_path: str = "knowledge_graph.json", model: str = "gpt-4o"):
        self.kg_path = kg_path
        self.model = model
        self.graph = None
        self.client = None
        self.entity_cache = {}
        self.frequent_queries = {
            "課程要求": ["學士學位課程", "修讀年期及畢業學分", "畢業資格"],
            "學生政策": ["學生紀律守則", "學生", "平等機會政策"],
            "考試規定": ["考試", "評估或考核", "補考"],
            "研究倫理": ["研究道德準則", "研究者", "研究參與者"],
            "AI工具": ["生成式 AI 工具", "學業不誠實行為"],
            "實習相關": ["實習", "實習單位", "有關學生實習支援"]
        }
        self.load_knowledge_graph()
        self.setup_openai()
    
    def setup_openai(self):
        """Initialize OpenAI client with compatibility handling"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            if OPENAI_V1:
                # Modern OpenAI v1+ API
                self.client = OpenAI(api_key=api_key)
                logger.info(f"✅ OpenAI v1+ client initialized with model: {self.model}")
            else:
                # Legacy OpenAI API
                openai.api_key = api_key
                self.client = openai
                logger.info(f"✅ Legacy OpenAI client initialized with model: {self.model}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise ValueError(f"Cannot initialize OpenAI client: {e}")
    
    def load_knowledge_graph(self):
        """Load knowledge graph from JSON file"""
        try:
            with open(self.kg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.graph = json_graph.node_link_graph(data)
            logger.info(f"✅ Knowledge graph loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            
            # Store metadata
            self.metadata = data.get('metadata', {})
            
            # Build entity cache for faster searching
            self._build_entity_cache()
            
        except FileNotFoundError:
            logger.error(f"❌ Knowledge graph file not found: {self.kg_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading knowledge graph: {e}")
            raise
    
    def _build_entity_cache(self):
        """Build cache for faster entity searching"""
        self.entity_cache = {
            'by_keywords': {},
            'entity_info': {}
        }
        
        for node in self.graph.nodes(data=True):
            entity_name = node[0]
            entity_data = node[1]
            
            # Store entity information
            self.entity_cache['entity_info'][entity_name] = {
                'name': entity_name,
                'type': entity_data.get('type', 'entity'),
                'size': entity_data.get('size', 5),
                'connections': len(list(self.graph.neighbors(entity_name)))
            }
    
    def search_entities(self, query: str, max_results: int = 10) -> List[Dict]:
        """Enhanced entity search with scoring"""
        query_lower = query.lower()
        matches = []
        
        for entity in self.graph.nodes():
            score = 0
            
            # Exact match
            if query_lower == entity.lower():
                score = 100
            # Contains query
            elif query_lower in entity.lower():
                score = 80
            # Character overlap for Chinese
            elif self._calculate_character_overlap(query, entity) > 0.5:
                score = 60
            # Keyword matches
            elif any(keyword in entity.lower() for keyword in query_lower.split()):
                score = 40
            
            if score > 0:
                matches.append({
                    'entity': entity,
                    'score': score,
                    'info': self.entity_cache['entity_info'].get(entity, {})
                })
        
        # Sort by score and return top results
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:max_results]
    
    def _calculate_character_overlap(self, query: str, entity: str) -> float:
        """Calculate character overlap ratio for Chinese text"""
        query_chars = set(query)
        entity_chars = set(entity)
        
        if not query_chars:
            return 0
        
        overlap = len(query_chars.intersection(entity_chars))
        return overlap / len(query_chars)
    
    def get_entity_relationships(self, entity: str) -> Dict:
        """Get relationships for a specific entity"""
        if entity not in self.graph:
            return {}
        
        relationships = {
            'outgoing': [],  # entity -> other
            'incoming': [],  # other -> entity
        }
        
        # Outgoing relationships
        for target in self.graph.successors(entity):
            edge_data = self.graph.get_edge_data(entity, target)
            for edge_attrs in edge_data.values():
                relationships['outgoing'].append({
                    'target': target,
                    'relation': edge_attrs.get('relation', 'related')
                })
        
        # Incoming relationships
        for source in self.graph.predecessors(entity):
            edge_data = self.graph.get_edge_data(source, entity)
            for edge_attrs in edge_data.values():
                relationships['incoming'].append({
                    'source': source,
                    'relation': edge_attrs.get('relation', 'related')
                })
        
        return relationships
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context from knowledge graph"""
        # Search for relevant entities
        entities = self.search_entities(query, 5)
        
        if not entities:
            return "沒有找到相關的知識圖譜信息。"
        
        context_parts = []
        context_parts.append("相關實體和關係:")
        
        for entity_info in entities:
            entity = entity_info['entity']
            score = entity_info['score']
            
            context_parts.append(f"\n【{entity}】(相關度: {score})")
            
            # Get relationships
            relationships = self.get_entity_relationships(entity)
            
            # Add outgoing relationships
            if relationships['outgoing']:
                context_parts.append("關係:")
                for rel in relationships['outgoing'][:5]:
                    context_parts.append(f"  • {entity} {rel['relation']} {rel['target']}")
            
            # Add incoming relationships
            if relationships['incoming']:
                context_parts.append("被關聯:")
                for rel in relationships['incoming'][:3]:
                    context_parts.append(f"  • {rel['source']} {rel['relation']} {entity}")
        
        return "\n".join(context_parts)
    
    def call_openai_api(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 1500) -> str:
        """Call OpenAI API with compatibility handling"""
        try:
            if OPENAI_V1:
                # Modern API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                # Legacy API
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return f"抱歉，處理您的問題時出現錯誤: {str(e)}"
    
    def generate_response(self, user_query: str) -> Dict:
        """Generate response using knowledge graph and GPT"""
        
        # Get context from knowledge graph
        kg_context = self.get_context_for_query(user_query)
        
        # Find relevant entities for metadata
        entities = self.search_entities(user_query, 5)
        
        # Prepare GPT prompt
        system_prompt = """你是澳門旅遊大學的智能助手，專門回答關於大學政策、課程和規定的問題。

請根據提供的知識圖譜信息回答用戶問題。回答要求：
1. 使用繁體中文
2. 基於知識圖譜提供準確信息
3. 結構清晰，易於理解
4. 如果信息不足，請誠實說明並建議聯繫相關部門
5. 保持專業和友好的語調"""

        user_prompt = f"""用戶問題: {user_query}

知識圖譜相關信息:
{kg_context}

請基於上述信息回答用戶的問題。"""

        # Generate response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        answer = self.call_openai_api(messages)
        
        return {
            'answer': answer,
            'relevant_entities': [e['entity'] for e in entities[:5]],
            'context_used': len(entities) > 0
        }

# Initialize the chatbot
kg_chatbot = None

@cl.on_chat_start
async def start():
    """Initialize the chatbot when chat starts"""
    global kg_chatbot
    
    try:
        # Show loading message
        loading_msg = cl.Message(content="🔄 正在初始化知識圖譜系統...")
        await loading_msg.send()
        
        # Initialize the knowledge graph chatbot
        kg_chatbot = CompatibleKnowledgeGraphChatbot()
        
        # Update loading message
        await loading_msg.update(content="✅ 系統初始化完成！")
        
        # Welcome message
        welcome_msg = f"""# 🎓 澳門旅遊大學智能助手

歡迎使用澳門旅遊大學智能知識圖譜助手！

## 📊 知識庫統計
- 🧠 **實體數量**: {len(kg_chatbot.graph.nodes)} 個
- 🔗 **關係數量**: {len(kg_chatbot.graph.edges)} 個  
- 📅 **最後更新**: {kg_chatbot.metadata.get('created', 'N/A')}
- 🤖 **AI模型**: {kg_chatbot.model}

## 🔍 我可以幫助您查詢

### 📚 學術相關
- 課程要求和畢業條件
- 評估方法和考試規定
- 學分計算和轉換

### 👨‍🎓 學生事務  
- 學生權利和責任
- 紀律守則和處分程序
- 申請流程和手續

### 🔬 研究支援
- 研究道德準則
- 學術誠信政策
- 實習安排和支援

### 🤖 AI工具使用
- 生成式AI工具政策
- 學術誠實行為規範

請輸入您的問題開始查詢！"""

        await cl.Message(content=welcome_msg).send()
        
        # Add quick action buttons
        actions = [
            cl.Action(name="課程要求", value="課程要求", description="查詢課程和畢業要求"),
            cl.Action(name="學生政策", value="學生政策", description="了解學生相關政策"),
            cl.Action(name="考試規定", value="考試規定", description="查看考試和評估規定"),
            cl.Action(name="研究倫理", value="研究倫理", description="研究道德和學術誠信"),
            cl.Action(name="AI工具", value="AI工具", description="生成式AI工具使用指引"),
        ]
        
        await cl.Message(
            content="🚀 **快速查詢**: 點擊下方按鈕快速查詢常見問題",
            actions=actions
        ).send()
        
    except Exception as e:
        error_msg = f"""❌ **初始化錯誤**

錯誤信息: {str(e)}

請檢查：
1. `knowledge_graph.json` 文件是否存在
2. OpenAI API 密鑰是否正確設定 (export OPENAI_API_KEY=sk-...)
3. OpenAI 庫版本 (嘗試: pip install openai>=1.0.0)
4. 網絡連接是否正常

**故障排除建議:**
```bash
# 檢查 API 密鑰
echo $OPENAI_API_KEY

# 更新 OpenAI 庫
pip install --upgrade openai

# 檢查文件
ls -la knowledge_graph.json
```

您可以嘗試重新開始對話。"""
        await cl.Message(content=error_msg).send()

@cl.action_callback("課程要求")
async def query_course_requirements(action):
    """Handle course requirements query"""
    await process_query("請告訴我關於學士學位課程的要求和畢業條件")

@cl.action_callback("學生政策")
async def query_student_policies(action):
    """Handle student policies query"""
    await process_query("學生紀律守則和相關政策有哪些？")

@cl.action_callback("考試規定")
async def query_exam_regulations(action):
    """Handle exam regulations query"""
    await process_query("考試和評估的相關規定是什麼？")

@cl.action_callback("研究倫理")
async def query_research_ethics(action):
    """Handle research ethics query"""
    await process_query("研究道德準則和學術誠信政策")

@cl.action_callback("AI工具")
async def query_ai_tools(action):
    """Handle AI tools query"""
    await process_query("生成式AI工具的使用政策和指引")

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    await process_query(message.content)

async def process_query(user_query: str):
    """Process user query"""
    global kg_chatbot
    
    if not kg_chatbot:
        await cl.Message(content="❌ 系統未初始化，請重新開始對話。").send()
        return
    
    # Show processing steps
    async with cl.Step(name="🔍 搜索知識圖譜", type="tool") as search_step:
        search_step.output = "正在搜索相關實體..."
        
        # Search for relevant entities
        entities = kg_chatbot.search_entities(user_query, 5)
        
        if entities:
            entity_list = [f"{e['entity']} (相關度: {e['score']})" for e in entities]
            search_step.output = f"✅ 找到 {len(entities)} 個相關實體:\n" + "\n".join([f"• {e}" for e in entity_list])
        else:
            search_step.output = "⚠️ 未找到直接匹配的實體，將進行智能分析..."
    
    async with cl.Step(name="🤖 生成回答", type="llm") as response_step:
        response_step.output = "正在使用 AI 生成智能回答..."
        
        # Generate response
        result = kg_chatbot.generate_response(user_query)
        
        response_step.output = "✅ 回答已生成"
    
    # Send main response
    response_content = result['answer']
    
    # Add entity information if available
    if result['relevant_entities']:
        response_content += f"\n\n## 📋 相關實體\n"
        for entity in result['relevant_entities']:
            response_content += f"• {entity}\n"
    
    await cl.Message(content=response_content).send()

@cl.on_stop
async def stop():
    """Clean up when chat stops"""
    goodbye_msg = """👋 **感謝使用澳門旅遊大學智能助手！**

希望我的回答對您有所幫助。如需更多信息，請：

📞 **聯繫教務部**: 查詢學術相關問題
🏫 **訪問官方網站**: 獲取最新政策更新  
📧 **發送郵件**: 獲取具體案例協助

祝您學習順利！🎓"""
    
    await cl.Message(content=goodbye_msg).send()

if __name__ == "__main__":
    print("🚀 Starting Compatible Chainlit Knowledge Graph Chatbot...")
    print("🔧 OpenAI Compatibility: Automatic detection")
    print("📋 Run with: chainlit run fixed_chainlit_chatbot.py -w")
    print("🌐 Then open: http://localhost:8000")
