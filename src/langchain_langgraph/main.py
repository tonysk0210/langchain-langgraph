from dotenv import load_dotenv  # 用於載入環境變數

load_dotenv()  # 載入環境變數
import os  # 用於處理作業系統相關的任務
from langchain_core.messages import HumanMessage  # 用於表示人類的訊息
from langgraph.graph import MessagesState, StateGraph, END  # 用於表示對話狀態，建立流程圖，LangGraph 的結束標記

from node import run_agent_reasoning, tool_node  # 用於運行代理推理和工具節點

# 定義節點名稱，方便引用
AGENT_REASON = "agent_reason"
ACT = "act"
# 定義常數，表示取清單最後一項
LAST = -1


# 判斷是否繼續對話的函數，決定下一步要去哪裡
def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END  # 回傳 LangGraph 的結束標記
    return ACT  # 回傳 "act" 節點


flow = StateGraph(MessagesState)  # 創建一個對話狀態圖

flow.add_node(AGENT_REASON, run_agent_reasoning)  # 增加一個節點，名為 "agent_reason"，運行 "run_agent_reasoning" 函數
flow.add_node(ACT, tool_node)  # 增加一個節點，名為 "act"，運行 "tool_node" 函數
flow.set_entry_point(AGENT_REASON)  # 設定入口點為 "agent_reason" 節點

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT})  # 增加一個有条件的邊，從 "agent_reason" 節點到 "act" 節點，條件為 "should_continue" 函數的回傳值

flow.add_edge(ACT, AGENT_REASON)  # 增加一個邊，從 "act" 節點到 "agent_reason" 節點

app = flow.compile()  # 編譯對話狀態圖
app.get_graph().draw_mermaid_png(output_file_path="flow.png")  # 繪製對話狀態圖並保存為 PNG 檔案

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    # print(os.getenv("OPENAI_API_KEY"))

    # 呼叫 LangGraph 並傳入初始對話訊息
    res = app.invoke({"messages": [HumanMessage(content="東京的氣溫是多少？請列出氣溫，然後將它乘以三。")]})
    print(res["messages"][LAST].content)  # 輸出最後一項對話訊息的內容
