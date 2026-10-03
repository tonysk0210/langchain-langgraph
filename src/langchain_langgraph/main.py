# import os  # 使用 os 模組來獲取環境變數

from dotenv import load_dotenv

load_dotenv()  # 載入環境變數 from .env file

from langchain.agents import create_agent  # 匯入建立 agent 的函式
from langchain.tools import tool  # 匯入將函式轉為 LangChain 工具的裝飾器
from langchain_core.messages import HumanMessage  # 匯入表示使用者訊息的類別
from langchain_openai import ChatOpenAI  # 匯入 ChatOpenAI 聊天模型類別
from tavily import TavilyClient  # 匯入 Tavily 搜尋用戶端類別

tavily = TavilyClient()


# @tool 會把這個函式轉成 LLM 可以呼叫的工具，LLM 收到的「工具說明書」如下：
#   - name        ← 函式名稱 "search"
#   - description ← 整段 docstring（原封不動，包含 Args / Returns 文字）
#   - parameters  ← 型別提示：query 必須是 string
# LLM 看不到函式內部的程式碼，只能靠上面三項判斷「何時使用」和「傳什麼參數」。
# 注意：這些 # 註解不會送給 LLM，只有 docstring 會。
@tool
def search(query: str) -> dict:
    """
    在網路上搜尋資訊的工具
    Args:
        query: 要搜尋的關鍵字
    Returns:
        搜尋結果
    """
    print(f"正在搜尋：{query}")  # LLM 決定呼叫此工具時，會自己產生 query，例如 "東京 天氣"
    return tavily.search(query=query)  # 這個回傳值會被送回 LLM，LLM 再根據它整理出最終回答

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)  # 創建 ChatOpenAI 物件，這個物件會呼叫 OpenAI API 來產生 LLM 回答
tools = [search]  # 把 search 工具放進 tools list
agent = create_agent(model=llm, tools=tools)  # 創建 agent，並把 llm 與 tools 傳入


def main() -> None:
    # print(os.getenv("OPENAI_API_KEY"))
    # print(os.getenv("TAVILY_API_KEY"))
    print("Hello from langchain-langgraph!")

    # 這裡的 agent.invoke() 會把 HumanMessage 送給 LLM，LLM 會判斷要不要呼叫 search 工具，並把結果整理成最終回答。
    result = agent.invoke({
        "messages": [HumanMessage(content="請幫我搜尋東京的天氣")]
    })
    print(f"搜尋結果：{result}")


main()
