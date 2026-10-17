from dotenv import load_dotenv  # 用於載入環境變數

load_dotenv()  # 載入環境變數
from langchain_core.tools import tool  # 用於定義工具
from langchain_openai import ChatOpenAI  # 用於與 OpenAI 進行互動
from langchain_tavily import TavilySearch  # 用於與 Tavily 進行互動


@tool
def triple(num: float) -> float:
    """
    參數 num：要乘以 3 的數字
    回傳：輸入數字的 3 倍
    """
    return float(num) * 3


# 工具實作與清單：這裡放的是 Python 真正能執行的工具。
# 同一份 tools 會交給 llm.bind_tools，並在 node.py 傳給 ToolNode。
tools = [TavilySearch(max_results=1), triple]

# 模型端的工具宣告：讓 LLM 知道可用工具的名稱、參數和用途。
# 模型呼叫時可以回傳 tool_calls（工具呼叫請求），但 bind_tools 不會執行工具本身。
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
