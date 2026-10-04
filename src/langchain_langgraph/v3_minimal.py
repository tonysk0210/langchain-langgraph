"""create_agent 最小版：不可再減的四樣東西 —— 工具、模型、create_agent、invoke。

跟 v1（原始 Ollama SDK 手刻 agent loop）做的事完全一樣，但沒有迴圈、沒有
tools_dict、沒有訊息維護、沒有終止判斷。那些全部在 LangGraph 裡面。
"""

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

SYSTEM_PROMPT = (
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


# docstring 會變成送進 LLM 的 tool description，不是普通註解。
@tool
def get_product_price(product_name: str) -> float:
    """根據產品名稱獲取產品價格。"""
    return {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}.get(product_name, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """根據價格和折扣等級計算折扣後的價格。Available discount tiers: 'gold', 'silver', 'bronze'."""
    return round(price * (1 - {"gold": 23, "silver": 12, "bronze": 5}.get(discount_tier, 0) / 100), 2)


agent = create_agent(
    model=init_chat_model("ollama:qwen3:1.7b", temperature=0),
    tools=[get_product_price, apply_discount],
    system_prompt=SYSTEM_PROMPT,
)

# messages 用原始 dict 即可，不必包 HumanMessage。
result = agent.invoke({
    "messages": [{"role": "user", "content": "請問 laptop 的價格是多少？如果我有 gold 會員，折扣後的價格是多少？"}]
})

print(result["messages"][-1].content)  # 列印最後一則訊息的內容
