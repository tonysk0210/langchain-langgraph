from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 定義評估推文的提示
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        # 系統提示
        (
            "system",
            "你是一位擅長讓推文爆紅的 Twitter 意見領袖，負責評估推文。請針對使用者的推文提出評論與建議。"
            "請務必提供詳細建議，包括長度、傳播力、風格等方面的調整方向。",
        ),
        MessagesPlaceholder(variable_name="messages"),  # 輸入的對話訊息
    ]
)

# 定義生成推文的提示
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位科技領域的 Twitter 意見領袖助理，負責撰寫優質推文。"
            "請根據使用者的需求，盡可能寫出最好的推文。"
            "如果使用者提供了評論，請根據評論修改先前的版本。",
        ),
        MessagesPlaceholder(variable_name="messages"),  # 輸入的對話訊息
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
generate_chain = generation_prompt | llm  # 生成推文的鏈
reflect_chain = reflection_prompt | llm  # 評估推文的鏈

