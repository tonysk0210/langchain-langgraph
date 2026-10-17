from dotenv import load_dotenv  # 用於載入環境變數

load_dotenv()  # 載入環境變數

from typing import TypedDict, Annotated  # 用於定義對話圖的類型
from langchain_core.messages import BaseMessage, HumanMessage  # 用於定義對話訊息
from langgraph.graph.message import add_messages  # 用於添加對話訊息
from langgraph.graph import END, StateGraph  # 用於定義對話圖

from chain2 import generate_chain, reflect_chain


# 定義對話圖的類型
class MessageGraph(TypedDict):  # TypedDict 的作用是描述字典預期有哪些 key，以及每個 key 的值應該是什麼型別，幫助 IDE 和型別檢查工具提早發現錯誤。
    messages: Annotated[list[BaseMessage], add_messages]  # 輸入的對話訊息 Annotated[欄位型別, LangGraph 使用的更新規則]


REFLECT = "reflect"
GENERATE = "generate"


# 定義生成節點
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


# 定義反射節點
def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}  # 輸出的對話訊息 as HumanMessage


# 定義是否繼續對話的節點
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder = StateGraph(state_schema=MessageGraph)  # 定義對話圖的類型 MessageGraph
builder.add_node(GENERATE, generation_node)  # 添加生成節點
builder.add_node(REFLECT, reflection_node)  # 添加反射節點
builder.set_entry_point(GENERATE)  # 設定起始節點為生成節點

builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})  # 添加條件邊，根據 should_continue 的回傳值決定是否繼續對話；第一個參數是起始節點，第二個參數是條件函數
builder.add_edge(REFLECT, GENERATE)  # 添加邊，從反射節點到生成節點

graph = builder.compile()  # 編譯對話圖
graph.get_graph().draw_mermaid_png(output_file_path="builder.png")  # 繪製對話狀態圖並保存為 PNG 檔案
# print(graph.get_graph().draw_mermaid())  # 繪製對話圖
# graph.get_graph().print_ascii()  # 印出對話圖

if __name__ == "__main__":
    print("Hello ReAct LangGraph with reflection")

    inputs = {
        "messages": [
            HumanMessage(
                content="""請幫我改進這則推文：
                                @LangChainAI
            — 全新的工具呼叫功能真的被嚴重低估了。

            等了好久，這項功能終於推出了——它讓我們能在不同模型上透過函式呼叫來實作代理，變得非常簡單。

            我製作了一支影片，介紹他們最新發布的部落格文章。

                                  """
            )
        ]
    }
    response = graph.invoke(inputs) # 呼叫對話圖
    print(response)
