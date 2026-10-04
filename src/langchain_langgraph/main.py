import json
import re
import inspect
from dotenv import load_dotenv

load_dotenv()  # 載入環境變數 from .env file

import ollama  # 匯入 ollama 模組，用於與 Ollama API 互動
from langsmith import traceable  # 匯入 traceable 裝飾器，用於追蹤函式的執行

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- 1. 工具實作：由 Python 執行，不是由模型直接執行 ---
# 模型只會從 prompt 中得知工具規格，並輸出要呼叫哪個工具及其參數；
# 真正執行下方 Python 函式的是後面的 run_agent。
@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """根據產品名稱獲取產品價格。"""
    # 這裡可以實現實際的價格查詢邏輯
    print(f"    >> 執行產品價格查詢（get_product_price）：{product_name}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product_name, 0.0)  # 如果產品名稱不存在，返回 0.0


# 依價格和折扣等級計算折後價格。
@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """根據價格和折扣等級計算折扣後的價格。Available discount tiers: 'gold', 'silver', 'bronze'."""
    print(f"    >> 執行折扣計算（apply_discount）：價格={price}, 折扣等級={discount_tier}")
    discount_percentages = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discount_percentages.get(discount_tier, 0.0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# --- 2. 把工具規格轉成 prompt 可讀的文字 ---
# Prompt-based tool calling 沒有使用模型 API 的原生 tool-call 機制：
# 我們把函式名稱、參數簽名和說明寫進 prompt，要求模型照指定格式輸出。
def get_tool_descriptions(tools_dict):
    descriptions = []  # 收集各工具說明
    for tool_name, tool_function in tools_dict.items():
        # 取得裝飾前的函式簽章，避免看到裝飾器包裝參數
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)  # 提供模型可用工具及其參數格式
tool_names = ", ".join(tools.keys())  # 限定 Action 可填入的工具名稱

# --- 3. ReAct prompt：用指示和文字格式引導模型選工具 ---
# Thought / Action / Action Input / Observation 是 prompt 約定的文字協定：
# 模型輸出 Action 與 JSON 參數；程式執行工具後，再把 Observation 餵回模型。
react_prompt = f"""
嚴格規則：
1. 回答產品價格或折扣問題時，必須使用工具。
2. 不得捏造價格或折扣。
3. 使用 get_product_price 取得產品價格後，必須呼叫 apply_discount。
4. 傳給 apply_discount 的價格必須與查得的價格完全相同，不得四捨五入或修改。
5. 不得將捏造的價格傳給 apply_discount。
6. 如果使用者未提供折扣等級，請詢問使用者折扣等級。不要假設。

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    # 只負責把目前對話內容送給 Ollama，並回傳模型回應。
    response = ollama.chat(model=MODEL, messages=messages, options=options)
    return response


# --- 4. Agent 控制流程：模型提議，程式解析並執行工具 ---
@traceable(name="run_agent-ReAct")
def run_agent(question: str):
    print(f"Question : {question}")
    print("=" * 60)

    # 初始 prompt 包含規則、工具清單、輸出格式和使用者問題。
    prompt = react_prompt.format(question=question)
    # scratchpad 保存前幾輪的模型輸出與工具結果，作為後續模型呼叫的上下文。
    scratchpad = ""

    # --- Agent loop ---
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # 每一輪都把工具規則和先前的 Thought/Action/Observation 一起交給模型。
        full_prompt = prompt + scratchpad

        ollama_response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["Observation:"], "temperature": 0},
        )  # 呼叫聊天模型並取得模型回應
        model_response = ollama_response.message.content  # 將模型回應存入 model_response 變數
        print(f"Model Response: {model_response}")

        # 先判斷模型是否已輸出最終答案；若有，就結束 agent loop。
        print(f"  >> [Parsing] 尋找 Final Answer in LLM model_response...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", model_response, re.IGNORECASE)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  >> [Parsed] 找到 Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer  # 如果模型回應中包含最終答案，則返回該答案

        # 否則把模型的文字回應當作「工具呼叫請求」解析。
        # 這一步是依賴 prompt 格式的文字解析，模型若格式不符就無法呼叫工具。
        print(f"  >> [Parsing] 尋找 Action and Action Input in LLM model_response...")
        action_match = re.search(r"Action:\s*(.+)", model_response)
        action_input_match = re.search(r"Action Input:\s*(.+)", model_response)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: 無法解析模型回應中的 Action 或 Action Input。請檢查模型回應格式。"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        # Action Input 必須是 JSON object，之後會以關鍵字參數形式傳給 Python 函式。
        print(
            f"    >> [Tool Selected] 準備執行工具：{tool_name}，"
            f"參數：{tool_input_raw}"
        )

        # 工具名稱需存在於本地 tools 字典；模型不能自行執行程式碼。
        # 程式負責驗證 JSON 形狀、呼叫對應函式，並將結果轉成文字。
        print(f"    >> [Tool Executing] 執行工具 {tool_name}...")
        if tool_name not in tools:
            observation = f"Error: 工具 {tool_name} 不存在。"
        else:
            try:
                tool_arguments = json.loads(tool_input_raw)
            except json.JSONDecodeError as error:
                observation = f"Error: Action Input 不是有效的 JSON：{error}"
            else:
                if not isinstance(tool_arguments, dict):
                    observation = "Error: Action Input 必須是 JSON object。"
                else:
                    observation = str(tools[tool_name](**tool_arguments))

        print(f"    >> [Tool Result] 工具 {tool_name} 執行結果：{observation}")
        # 把模型剛才的 Action 和程式產生的 Observation 加回上下文；
        # 下一輪模型便能根據真實工具結果決定下一步或產生 Final Answer。
        scratchpad += f"\n{model_response}\nObservation: {observation}\nThought:"

    print("\n已達到最大迭代次數，仍未得到最終答案。")
    return None


if __name__ == "__main__":
    print("Hello from langchain-langgraph! (.bind_tools)")
    print()
    result = run_agent("請問 laptop 的價格是多少？如果我有 gold 會員，折扣後的價格是多少？")
