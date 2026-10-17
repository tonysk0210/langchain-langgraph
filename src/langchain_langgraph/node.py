from dotenv import load_dotenv  # 用於載入環境變數

load_dotenv()  # 載入環境變數
from langgraph.graph import MessagesState  # graph 傳遞的狀態；messages 保存對話訊息
from langgraph.prebuilt import ToolNode  # LangGraph 預建的工具執行節點

from react import llm, tools  # 重用 react.py 準備好的模型與可執行工具清單

SYSYEM_MESSAGE = """
你是一位樂於助人的助理，可以使用工具回答問題。
"""


# 模型節點：讀取目前對話，請模型決定要直接回答或提出工具呼叫。
def run_agent_reasoning(state: MessagesState) -> MessagesState:  # MessagesState 是一個類別，用於管理訊息狀態
    """
    讀取對話狀態並取得模型回覆；若回覆含 tool_calls，這裡只提出請求，尚未執行工具。
    """
    # 把 system 指示和既有對話一起送給模型；response 是模型產生的 AIMessage。
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    # 回傳 state 更新：messages 欄位用清單裝新訊息，LangGraph 會依 reducer 合併到對話狀態。
    return {"messages": [response]}


# 工具執行節點：接收含 tool_calls 的最新 AIMessage，依工具名稱和參數執行 tools 中的 Python 工具。
# 執行結果會以 ToolMessage 回傳到 messages；此處只建立節點，仍需在 StateGraph 中加入並設定路由才會執行。
tool_node = ToolNode(tools)

# MessageState 的結構
# state = {
#     "messages": [
#         {"role": "user", "content": "你好"}
#     ]
# }

# response 的結構
# AIMessage(
#     content="",
#     tool_calls=[
#         {
#             "name": "triple",
#             "args": {"num": 5},
#             "id": "call_...",
#             "type": "tool_call",
#         }
#     ],
# )

# llm.invoke([
#     {"role": "system", "content": SYSYEM_MESSAGE},  # 字典
#     HumanMessage(content="你好"),                    # 訊息物件
# ])
