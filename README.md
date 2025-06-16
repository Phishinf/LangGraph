# README.md
# 澳門旅遊大學知識圖譜智能助手

基於知識圖譜和 GPT-4o 的智能查詢系統，可以回答關於澳門旅遊大學政策、課程和規定的問題。

## 功能特點

- 🧠 **知識圖譜驅動**: 基於從大學手冊提取的結構化知識
- 🤖 **GPT-4o 智能分析**: 自然語言理解和生成
- 🔍 **實體搜索**: 快速找到相關的大學實體和概念
- 📊 **關係分析**: 分析實體之間的複雜關係
- 🎯 **精準回答**: 基於官方政策提供準確信息

## 安裝步驟

1. 克隆或下載項目文件
2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

3. 設置 OpenAI API 密鑰：
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   # 或者創建 .env 文件
   ```

4. 確保 `knowledge_graph.json` 文件在同一目錄

5. 運行應用：
   ```bash
   chainlit run chainlit_kg_chatbot.py -w
   ```

## 使用方式

1. 打開瀏覽器訪問 `http://localhost:8000`
2. 輸入關於澳門旅遊大學的問題
3. 系統會搜索知識圖譜並生成智能回答

## 支持的查詢類型

- 學術政策和規定
- 課程要求和評估
- 學生權利和責任
- 研究道德準則
- AI工具使用政策
- 大學服務和支援

## 技術架構

- **前端**: Chainlit Web UI
- **後端**: Python + NetworkX
- **AI**: OpenAI GPT-4o
- **知識庫**: JSON格式的知識圖譜
