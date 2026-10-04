from typing import Any

from dotenv import load_dotenv
from pydantic_core.core_schema import plain_serializer_function_ser_schema

load_dotenv()  # 載入環境變數 from .env file

from langchain.chat_models import init_chat_model  # 匯入自訂的 init_chat_model 函式，用於初始化聊天模型
from langchain.tools import tool  # 匯入 tool 裝飾器，用於定義工具函式
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # 匯入 HumanMessage、SystemMessage、ToolMessage 類別，用於定義不同類型的訊息
from langsmith import traceable  # 匯入 traceable 裝飾器，用於追蹤函式的執行

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools ---
# 依產品名稱查詢價格；找不到產品時回傳 0.0。
@tool
def get_product_price(product_name: str) -> float:
    """根據產品名稱獲取產品價格。"""
    # 這裡可以實現實際的價格查詢邏輯
    print(f"    >> 執行產品價格查詢（get_product_price）：{product_name}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product_name, 0.0)  # 如果產品名稱不存在，返回 0.0


# 依價格和折扣等級計算折後價格；不支援的等級不打折。
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """根據價格和折扣等級計算折扣後的價格。Available discount tiers: 'gold', 'silver', 'bronze'."""
    print(f"    >> 執行折扣計算（apply_discount）：價格={price}, 折扣等級={discount_tier}")
    discount_percentages = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discount_percentages.get(discount_tier, 0.0)
    return round(price * (1 - discount / 100), 2)


# --- Agent ---
@traceable(name="run_agent")
def run_agent(question: str) -> None | str | list[str | dict[Any, Any]]:
    tools = [get_product_price, apply_discount]  # 定義可用的工具列表
    tools_dict = {tool.name: tool for tool in tools}  # 將工具列表轉換為字典，方便查找

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)  # 初始化聊天模型
    llm_with_tools = llm.bind_tools(tools)  # 將工具綁定到聊天模型

    print(f"Question : {question}")
    print("=" * 60)

    # 建立訊息列表，包含系統訊息和使用者訊息
    messages = [
        SystemMessage(
            content=(
                "你是一位樂於助人的購物助理。\n"
                "你可以使用以下工具：get_product_price、apply_discount。\n\n"
                "嚴格規則：\n"
                "1. 回答產品價格或折扣問題時，必須使用工具。\n"
                "2. 不得捏造價格或折扣。\n"
                "3. 使用 get_product_price 取得產品價格後，必須呼叫 apply_discount。\n"
                "4. 傳給 apply_discount 的價格必須與查得的價格完全相同，不得四捨五入或修改。\n"
                "5. 不得將捏造的價格傳給 apply_discount。\n"
                "6. 如果使用者未提供折扣等級，請詢問使用者折扣等級。不要假設。"
            ),
        ),
        HumanMessage(content=question),
    ]  # 初始化訊息列表，包含系統訊息和使用者訊息

    # --- Agent loop ---
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        model_response = llm_with_tools.invoke(messages)  # 呼叫聊天模型並取得模型回應
        tool_call_requests = model_response.tool_calls  # 取得模型提出的工具呼叫請求

        # 1. 如果有工具呼叫，則執行工具並將結果加入訊息列表
        if not tool_call_requests:
            print(f"\n Final Answer: {model_response.content}")  # 如果沒有工具呼叫，表示 AI 已經給出最終答案，打印最終答案並返回
            return model_response.content

        # 2. 保留模型提出的工具呼叫，讓下一輪訊息包含完整脈絡
        messages.append(model_response)

        # 3. 取得第一個工具呼叫請求，拆出工具名稱、參數和呼叫 ID
        tool_call_request = tool_call_requests[0]
        requested_tool_name = tool_call_request.get("name")
        tool_arguments = tool_call_request.get("args", {})
        tool_call_id = tool_call_request.get("id")

        print(
            f"    >> [Tool Selected] 準備執行工具：{requested_tool_name}，"
            f"參數：{tool_arguments}，呼叫 ID：{tool_call_id}"
        )

        # 4. 根據工具名稱選擇對應的工具
        selected_tool = tools_dict.get(requested_tool_name)
        if selected_tool is None:
            raise ValueError(f"找不到工具：{requested_tool_name}")

        # 5. 執行工具呼叫，並將結果加入訊息列表
        tool_result = selected_tool.invoke(tool_call_request)  # 執行模型提出的工具呼叫
        print(f"    >> [Tool Result] 工具執行結果：{tool_result.content}")
        messages.append(tool_result)  # 把工具結果交回模型，供下一輪使用

    print("\n已達到最大迭代次數，仍未得到最終答案。")
    return None


if __name__ == "__main__":
    print("Hello from langchain-langgraph! (.bind_tools)")
    print()
    result = run_agent("請問 laptop 的價格是多少？如果我有 gold 會員，折扣後的價格是多少？")
