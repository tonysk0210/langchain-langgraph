# import os  # 使用 os 模組來獲取環境變數

from dotenv import load_dotenv

load_dotenv()  # 載入環境變數 from .env file

from typing import List
from pydantic import BaseModel, Field  # 匯入 Pydantic 的 BaseModel 與 Field 類別，用於定義資料模型
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


# Source 和 AgentResponse 的類別 docstring：提供模型的整體用途說明。
# Field(description=...)：描述個別欄位，例如 url、answer、sources。

class Source(BaseModel):  # 定義 Source 類別，繼承自 Pydantic 的 BaseModel
    """一筆供回答引用的網頁來源。"""
    url: str = Field(description="該來源網頁的完整網址")  # 定義 url 欄位，型別為字串，並加上描述


class AgentResponse(BaseModel):  # 定義 AgentResponse 類別，繼承自 Pydantic 的 BaseModel
    """包含回答內容及其參考來源的結構化回應。"""
    answer: str = Field(description="根據搜尋結果整理的回答")  # 定義 answer 欄位，型別為字串，並加上描述
    sources: List[Source] = Field(
        default_factory=list,
        description="回答實際參考的網頁來源；每筆來源包含完整網址",
    )  # 定義 sources 欄位，型別為 Source 的列表，預設為空列表，並加上描述


llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main() -> None:
    print("Hello from langchain-langgraph!")

    # 這裡的 agent.invoke() 會把 HumanMessage 送給 LLM，LLM 會判斷要不要呼叫 search 工具，並把結果整理成最終回答。
    result = agent.invoke({
        "messages": [HumanMessage(content="請幫我搜尋東京的天氣")]
    })
    print(f"搜尋結果：{result}")


main()
