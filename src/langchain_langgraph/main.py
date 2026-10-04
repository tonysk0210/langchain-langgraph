"""用 Ollama SDK 手刻 agent loop（對照組：上一版的 LangChain `.bind_tools`）。

兩版送給模型的 tool schema 完全相同，差異只在抽象層。
以下 Difference #1~#8 標出 LangChain 幫你做掉、這裡要自己寫的部分。
"""

from typing import Any
from dotenv import load_dotenv

load_dotenv()  # 載入環境變數 from .env file

import ollama  # 匯入 ollama 模組，用於與 Ollama API 互動
from langsmith import traceable  # 匯入 traceable 裝飾器，用於追蹤函式的執行

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools ---
# Difference #1：工具註冊
#   @tool（產生 schema + 自動 tracing） → @traceable（只有 tracing）
#   所以 docstring 不會進 prompt，真正的描述在下方 tools_for_llm。
# 依產品名稱查詢價格。
@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """根據產品名稱獲取產品價格。"""
    # 這裡可以實現實際的價格查詢邏輯
    print(f"    >> 執行產品價格查詢（get_product_price）：{product_name}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product_name, 0.0)  # 如果產品名稱不存在，返回 0.0


# 依價格和折扣等級計算折後價格；不支援的等級不打折。
@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """根據價格和折扣等級計算折扣後的價格。Available discount tiers: 'gold', 'silver', 'bronze'."""
    print(f"    >> 執行折扣計算（apply_discount）：價格={price}, 折扣等級={discount_tier}")
    discount_percentages = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discount_percentages.get(discount_tier, 0.0)
    return round(price * (1 - discount / 100), 2)


# Difference #2：schema 來源
#   簽章 + docstring 自動產生 → 手寫 JSON Schema（內容與上一版等價）
#   代價：改了函式簽章，這裡不會跟著改。
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "根據產品名稱獲取產品價格。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "根據價格和折扣等級計算折扣後的價格。Available discount tiers: 'gold', 'silver', 'bronze'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                    },
                    "discount_tier": {
                        "type": "string",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# Difference #3：模型與工具綁定
#   init_chat_model + .bind_tools（可換 provider、工具記住） → ollama.chat 寫死，每輪自己帶 tools
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """使用 Ollama API 呼叫聊天模型，並追蹤訊息。"""
    print(f"    >> 呼叫 Ollama API，訊息數量：{len(messages)}")
    response = ollama.chat(
        MODEL,
        messages=messages,
        tools=tools_for_llm,  # ← 每一輪都要重新傳，沒有 bind_tools 幫你記住
        options={"temperature": 0},
    )  # 呼叫 Ollama API，傳入訊息、工具列表和模型參數
    return response


# --- Agent ---
@traceable(name="run_agent")
def run_agent(question: str) -> None | str | list[str | dict[Any, Any]]:
    # Difference #4：工具查找表
    #   {t.name: t for t in tools} → 手打字串 key，要跟 tools_for_llm 的 name 人工同步
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question : {question}")
    print("=" * 60)

    # Difference #5：訊息格式
    #   SystemMessage / HumanMessage → 原始 dict，看得到真實 payload，但沒有型別檢查
    # 建立訊息列表，包含系統訊息和使用者訊息
    messages = [
        {
            "role": "system",
            "content": (
                "你是一位樂於助人的購物助理。\n"
                "你可以使用以下工具：get_product_price、apply_discount。\n\n"
                "嚴格規則：\n"
                "1. 回答產品價格或折扣問題時，必須使用工具。\n"
                "2. 不得捏造價格或折扣。\n"
                "3. 使用 get_product_price 取得產品價格後，必須呼叫 apply_discount。\n"
                "4. 傳給 apply_discount 的價格必須與查得的價格完全相同，不得四捨五入或修改。\n"
                "5. 不得將捏造的價格傳給 apply_discount。\n"
                "6. 如果使用者未提供折扣等級，請詢問使用者折扣等級。不要假設。"
            )
        },
        {"role": "user", "content": question},
    ]  # 初始化訊息列表，包含系統訊息和使用者訊息

    # --- Agent loop ---
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference #6：回應形狀
        #   llm.invoke() → AIMessage（跨 provider 已正規化） → ChatResponse，要先 .message
        ollama_response = ollama_chat_traced(messages=messages)  # 呼叫聊天模型並取得模型回應
        model_response = ollama_response.message  # 將模型回應存入 model_response 變數

        tool_call_requests = model_response.tool_calls  # 取得模型提出的工具呼叫請求

        # 1. 如果有工具呼叫，則執行工具並將結果加入訊息列表
        if not tool_call_requests:
            print(f"\n Final Answer: {model_response.content}")  # 如果沒有工具呼叫，表示 AI 已經給出最終答案，打印最終答案並返回
            return model_response.content

        # 2. 保留模型提出的工具呼叫，讓下一輪訊息包含完整脈絡
        messages.append(model_response)

        # 3. 取得第一個工具呼叫請求，拆出工具名稱與參數
        #    （承 #6）上一版：tc.get("name") / tc.get("args") / tc.get("id")
        #            這一版：巢狀物件屬性，且沒有 id
        tool_call_request = tool_call_requests[0]
        requested_tool_name = tool_call_request.function.name
        tool_arguments = tool_call_request.function.arguments

        print(
            f"    >> [Tool Selected] 準備執行工具：{requested_tool_name}，"
            f"參數：{tool_arguments}"
        )

        # 4. 根據工具名稱選擇對應的工具
        selected_tool = tools_dict.get(requested_tool_name)
        if selected_tool is None:
            raise ValueError(f"找不到工具：{requested_tool_name}")

        # Difference #7：執行工具
        #   tool.invoke(tool_call)（自動解包 + Pydantic 驗型） → 自己 **kwargs 展開，無驗型
        tool_result = selected_tool(**tool_arguments)  # 執行模型提出的工具呼叫
        print(f"    >> [Tool Result] 工具執行結果：{tool_result}")

        # Difference #8：回填結果
        #   .invoke() 直接回傳 ToolMessage → 自己組 dict、自己 str() 轉型
        messages.append({
            "role": "tool",
            "tool_name": requested_tool_name,
            "content": str(tool_result),
        })  # 把工具結果交回模型，供下一輪使用

    print("\n已達到最大迭代次數，仍未得到最終答案。")
    return None


if __name__ == "__main__":
    print("Hello from langchain-langgraph! (.bind_tools)")
    print()
    result = run_agent("請問 laptop 的價格是多少？如果我有 gold 會員，折扣後的價格是多少？")
