# import os  # 使用 os 模組來獲取環境變數

from dotenv import load_dotenv

load_dotenv()  # 載入環境變數 from .env file

from langchain.agents import create_agent  # 匯入建立 agent 的函式
from langchain_core.messages import HumanMessage  # 匯入表示使用者訊息的類別
from langchain_openai import ChatOpenAI  # 匯入 ChatOpenAI 聊天模型類別
from langchain_tavily import TavilySearch  # 匯入 TavilySearch 工具類別

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)  # 創建 ChatOpenAI 物件，這個物件會呼叫 OpenAI API 來產生 LLM 回答
tools = [TavilySearch()]  # 創建 TavilySearch 工具物件，這個物件會呼叫 Tavily API 來搜尋網頁資訊
agent = create_agent(model=llm, tools=tools)  # 創建 agent，並把 llm 與 tools 傳入

def main() -> None:
    print("Hello from langchain-langgraph!")

    # 這裡的 agent.invoke() 會把 HumanMessage 送給 LLM，LLM 會判斷要不要呼叫 search 工具，並把結果整理成最終回答。
    result = agent.invoke({
        "messages": [HumanMessage(content="請幫我搜尋東京的天氣")]
    })
    print(f"搜尋結果：{result}")

main()
