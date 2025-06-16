import chainlit as cl
import json
import networkx as nx
from networkx.readwrite import json_graph
from openai import OpenAI
import os
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import asyncio
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedKnowledgeGraphChatbot:
    """Enhanced Chainlit chatbot with advanced knowledge graph capabilities"""
    
    def __init__(self, kg_path: str = "knowledge_graph.json", model: str = "gpt-4o"):
        self.kg_path = kg_path
        self.model = model
        self.graph = None
        self.client = None
        self.use_legacy_api = False  # Flag for API compatibility
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
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            # Initialize OpenAI client with minimal parameters
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            # Fallback: try with older initialization method
            try:
                import openai
                openai.api_key = api_key
                self.client = openai
                self.use_legacy_api = True
                logger.info("Using legacy OpenAI API")
            except Exception as e2:
                raise ValueError(f"Cannot initialize OpenAI client: {e}. Please update openai library: pip install openai>=1.0.0")
    
    def load_knowledge_graph(self):
        """Load knowledge graph from JSON file"""
        try:
            with open(self.kg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.graph = json_graph.node_link_graph(data)
            logger.info(f"Knowledge graph loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            
            # Store metadata
            self.metadata = data.get('metadata', {})
            
            # Build entity cache for faster searching
            self._build_entity_cache()
            
        except FileNotFoundError:
            logger.error(f"Knowledge graph file not found: {self.kg_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            raise
    
    def _build_entity_cache(self):
        """Build cache for faster entity searching"""
        self.entity_cache = {
            'by_keywords': {},
            'by_category': {},
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
            
            # Index by keywords
            keywords = self._extract_keywords(entity_name)
            for keyword in keywords:
                if keyword not in self.entity_cache['by_keywords']:
                    self.entity_cache['by_keywords'][keyword] = []
                self.entity_cache['by_keywords'][keyword].append(entity_name)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from entity name"""
        # Simple keyword extraction for Chinese and English
        keywords = []
        
        # Add full text
        keywords.append(text.lower())
        
        # Split by common separators
        parts = re.split(r'[、，。！？\s\-_/]+', text)
        for part in parts:
            if len(part) > 1:
                keywords.append(part.lower())
        
        # Extract individual Chinese characters for fuzzy matching
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        keywords.extend(chinese_chars)
        
        return list(set(keywords))
    
    def smart_search_entities(self, query: str, max_results: int = 10) -> List[Dict]:
        """Enhanced entity search with scoring"""
        query_lower = query.lower()
        matches = []
        
        # Exact matches
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
    
    def get_entity_context(self, entity: str, depth: int = 2) -> Dict:
        """Get comprehensive context for an entity"""
        if entity not in self.graph:
            return {}
        
        context = {
            'entity': entity,
            'direct_relationships': [],
            'related_policies': [],
            'relevant_procedures': [],
            'connected_services': []
        }
        
        # Get direct relationships
        for target in self.graph.successors(entity):
            edge_data = self.graph.get_edge_data(entity, target)
            for edge_attrs in edge_data.values():
                context['direct_relationships'].append({
                    'target': target,
                    'relation': edge_attrs.get('relation', 'related'),
                    'type': 'outgoing'
                })
        
        for source in self.graph.predecessors(entity):
            edge_data = self.graph.get_edge_data(source, entity)
            for edge_attrs in edge_data.values():
                context['direct_relationships'].append({
                    'source': source,
                    'relation': edge_attrs.get('relation', 'related'),
                    'type': 'incoming'
                })
        
        # Categorize relationships
        for rel in context['direct_relationships']:
            target_or_source = rel.get('target') or rel.get('source')
            if any(keyword in target_or_source for keyword in ['政策', '守則', '規範']):
                context['related_policies'].append(target_or_source)
            elif any(keyword in target_or_source for keyword in ['申請', '程序', '手續']):
                context['relevant_procedures'].append(target_or_source)
            elif any(keyword in target_or_source for keyword in ['部', '中心', '服務']):
                context['connected_services'].append(target_or_source)
        
        return context
    
    def suggest_related_queries(self, current_query: str) -> List[str]:
        """Suggest related queries based on current query"""
        suggestions = []
        
        # Check frequent query categories
        for category, entities in self.frequent_queries.items():
            if any(keyword in current_query for keyword in entities):
                suggestions.append(f"更多關於{category}的信息")
        
        # Find related entities
        entities = self.smart_search_entities(current_query, 5)
        for entity_info in entities[:3]:
            entity_name = entity_info['entity']
            neighbors = list(self.graph.neighbors(entity_name))
            for neighbor in neighbors[:2]:
                suggestions.append(f"{entity_name}與{neighbor}的關係")
        
        return list(set(suggestions))[:5]
    
    def generate_comprehensive_response(self, user_query: str) -> Dict:
        """Generate comprehensive response with multiple information sources"""
        
        # 1. Search for relevant entities
        relevant_entities = self.smart_search_entities(user_query, 8)
        
        # 2. Get context for top entities
        contexts = []
        for entity_info in relevant_entities[:5]:
            entity = entity_info['entity']
            context = self.get_entity_context(entity)
            if context:
                contexts.append(context)
        
        # 3. Prepare comprehensive knowledge base
        kg_context = self._format_comprehensive_context(contexts, relevant_entities)
        
        # 4. Generate suggestions
        suggestions = self.suggest_related_queries(user_query)
        
        # 5. Create GPT prompt
        system_prompt = """你是澳門旅遊大學的高級智能助手，專門提供準確、全面的大學信息諮詢服務。

你的職責：
1. 基於知識圖譜提供準確的政策解釋
2. 指導學生了解程序和要求
3. 解釋大學規定和標準
4. 提供相關服務和支援信息
5. 確保信息的時效性和準確性

回答指南：
- 使用繁體中文
- 提供結構化和詳細的回答
- 引用具體的政策或規定名稱
- 包含實用的操作建議
- 當信息不完整時，建議聯繫相關部門
- 保持專業和友好的語調"""

        user_prompt = f"""用戶查詢: {user_query}

知識圖譜相關信息:
{kg_context}

請提供全面、準確的回答，包括：
1. 直接回答用戶問題
2. 相關政策或規定的解釋
3. 具體的操作步驟（如適用）
4. 相關部門或服務的聯繫建議
5. 注意事項或重要提醒

請確保回答結構清晰，易於理解。"""

        try:
            # Call GPT-4o with version compatibility
            if hasattr(self, 'use_legacy_api') and self.use_legacy_api:
                # Legacy API call
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                answer = response.choices[0].message.content
            else:
                # Modern API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'relevant_entities': [e['entity'] for e in relevant_entities[:5]],
                'suggestions': suggestions,
                'context_used': len(contexts) > 0
            }
            
        except Exception as e:
            logger.error(f"Error calling GPT-4o: {e}")
            return {
                'answer': "抱歉，處理您的問題時出現錯誤。請稍後再試。",
                'relevant_entities': [],
                'suggestions': [],
                'context_used': False
            }
    
    def _format_comprehensive_context(self, contexts: List[Dict], entities: List[Dict]) -> str:
        """Format comprehensive context for GPT"""
        context_parts = []
        
        # Add entity overview
        if entities:
            context_parts.append("相關實體概覽:")
            for entity_info in entities[:5]:
                entity = entity_info['entity']
                score = entity_info['score']
                connections = entity_info['info'].get('connections', 0)
                context_parts.append(f"- {entity} (相關度: {score}, 連接數: {connections})")
        
        # Add detailed contexts
        for context in contexts:
            entity = context['entity']
            context_parts.append(f"\n【{entity}】詳細信息:")
            
            # Direct relationships
            if context['direct_relationships']:
                context_parts.append("關係:")
                for rel in context['direct_relationships'][:8]:
                    if rel['type'] == 'outgoing':
                        context_parts.append(f"  • {entity} {rel['relation']} {rel['target']}")
                    else:
                        context_parts.append(f"  • {rel['source']} {rel['relation']} {entity}")
            
            # Related policies
            if context['related_policies']:
                context_parts.append(f"相關政策: {', '.join(context['related_policies'][:3])}")
            
            # Procedures
            if context['relevant_procedures']:
                context_parts.append(f"相關程序: {', '.join(context['relevant_procedures'][:3])}")
            
            # Services
            if context['connected_services']:
                context_parts.append(f"相關服務: {', '.join(context['connected_services'][:3])}")
        
        return "\n".join(context_parts)

# Initialize the enhanced chatbot
enhanced_kg_chatbot = None

@cl.on_chat_start
async def start():
    """Initialize the enhanced chatbot when chat starts"""
    global enhanced_kg_chatbot
    
    try:
        # Show loading message
        loading_msg = cl.Message(content="🔄 正在初始化知識圖譜系統...")
        await loading_msg.send()
        
        # Initialize the knowledge graph chatbot
        enhanced_kg_chatbot = EnhancedKnowledgeGraphChatbot()
        
        # Update loading message
        await loading_msg.update(content="✅ 系統初始化完成！")
        
        # Welcome message with quick actions
        welcome_msg = f"""# 🎓 澳門旅遊大學智能助手 (增強版)

歡迎使用澳門旅遊大學智能知識圖譜助手！我已經準備好為您提供全面的查詢服務。

## 📊 知識庫統計
- 🧠 **實體數量**: {len(enhanced_kg_chatbot.graph.nodes)} 個
- 🔗 **關係數量**: {len(enhanced_kg_chatbot.graph.edges)} 個  
- 📅 **最後更新**: {enhanced_kg_chatbot.metadata.get('created', 'N/A')}

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

## 💡 使用提示
輸入您的問題，我會：
1. 🔍 搜索相關的知識實體
2. 📊 分析實體之間的關係
3. 🤖 生成準確的回答
4. 💡 提供相關建議

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
2. OpenAI API 密鑰是否正確設定
3. 網絡連接是否正常

您可以嘗試重新開始對話。"""
        await cl.Message(content=error_msg).send()

@cl.action_callback("課程要求")
async def query_course_requirements(action):
    """Handle course requirements query"""
    await handle_predefined_query("請告訴我關於學士學位課程的要求和畢業條件")

@cl.action_callback("學生政策")
async def query_student_policies(action):
    """Handle student policies query"""
    await handle_predefined_query("學生紀律守則和相關政策有哪些？")

@cl.action_callback("考試規定")
async def query_exam_regulations(action):
    """Handle exam regulations query"""
    await handle_predefined_query("考試和評估的相關規定是什麼？")

@cl.action_callback("研究倫理")
async def query_research_ethics(action):
    """Handle research ethics query"""
    await handle_predefined_query("研究道德準則和學術誠信政策")

@cl.action_callback("AI工具")
async def query_ai_tools(action):
    """Handle AI tools query"""
    await handle_predefined_query("生成式AI工具的使用政策和指引")

async def handle_predefined_query(query: str):
    """Handle predefined queries"""
    global enhanced_kg_chatbot
    
    if not enhanced_kg_chatbot:
        await cl.Message(content="❌ 系統未初始化，請重新開始對話。").send()
        return
    
    # Process the query
    await process_query(query)

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    await process_query(message.content)

async def process_query(user_query: str):
    """Process user query with enhanced features"""
    global enhanced_kg_chatbot
    
    if not enhanced_kg_chatbot:
        await cl.Message(content="❌ 系統未初始化，請重新開始對話。").send()
        return
    
    # Show detailed processing steps
    async with cl.Step(name="🔍 智能搜索", type="tool") as search_step:
        search_step.output = "正在搜索知識圖譜中的相關實體..."
        
        # Search for relevant entities
        entities = enhanced_kg_chatbot.smart_search_entities(user_query, 8)
        
        if entities:
            entity_list = [f"{e['entity']} (相關度: {e['score']})" for e in entities[:5]]
            search_step.output = f"✅ 找到 {len(entities)} 個相關實體:\n" + "\n".join([f"• {e}" for e in entity_list])
        else:
            search_step.output = "⚠️ 未找到直接匹配的實體，將進行智能分析..."
    
    async with cl.Step(name="📊 關係分析", type="tool") as analysis_step:
        analysis_step.output = "正在分析實體關係和獲取上下文信息..."
        
        # Get contexts
        contexts = []
        for entity_info in entities[:3]:
            context = enhanced_kg_chatbot.get_entity_context(entity_info['entity'])
            if context:
                contexts.append(context)
        
        if contexts:
            analysis_step.output = f"✅ 分析了 {len(contexts)} 個核心實體的關係網絡\n"
            for i, ctx in enumerate(contexts[:3], 1):
                rel_count = len(ctx.get('direct_relationships', []))
                analysis_step.output += f"• {ctx['entity']}: {rel_count} 個直接關係\n"
        else:
            analysis_step.output = "⚠️ 未找到具體的關係信息，將基於一般知識回答"
    
    async with cl.Step(name="🤖 生成回答", type="llm") as response_step:
        response_step.output = "正在使用 GPT-4o 生成智能回答..."
        
        # Generate comprehensive response
        result = enhanced_kg_chatbot.generate_comprehensive_response(user_query)
        
        response_step.output = "✅ 回答已生成，包含相關實體和建議"
    
    # Main response
    response_content = result['answer']
    
    # Add entity information if available
    if result['relevant_entities']:
        response_content += f"\n\n## 📋 相關實體\n"
        for entity in result['relevant_entities']:
            response_content += f"• {entity}\n"
    
    await cl.Message(content=response_content).send()
    
    # Add suggestions if available
    if result['suggestions']:
        suggestions_actions = []
        for i, suggestion in enumerate(result['suggestions'][:3]):
            suggestions_actions.append(
                cl.Action(
                    name=f"suggestion_{i}",
                    value=suggestion,
                    description=f"查詢: {suggestion}"
                )
            )
        
        await cl.Message(
            content="💡 **相關建議**: 您可能還想了解",
            actions=suggestions_actions
        ).send()

@cl.action_callback("suggestion_0")
async def handle_suggestion_0(action):
    await handle_predefined_query(action.value)

@cl.action_callback("suggestion_1") 
async def handle_suggestion_1(action):
    await handle_predefined_query(action.value)

@cl.action_callback("suggestion_2")
async def handle_suggestion_2(action):
    await handle_predefined_query(action.value)

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
    print("🚀 Starting Enhanced Chainlit Knowledge Graph Chatbot...")
    print("📋 Features:")
    print("   • Smart entity search with scoring")
    print("   • Comprehensive relationship analysis") 
    print("   • Query suggestions")
    print("   • Quick action buttons")
    print("   • Enhanced user interface")
    print("\n💡 Run with: chainlit run enhanced_chainlit_chatbot.py -w")
    print("🌐 Then open: http://localhost:8000")