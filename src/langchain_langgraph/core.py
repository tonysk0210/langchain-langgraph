import os  # 用於處理作業系統相關的任務
from typing import Any, Dict, List  # 用於型別提示

from dotenv import load_dotenv  # 用於載入環境變數

load_dotenv()  # 載入環境變數
from langchain_pinecone import PineconeVectorStore  # 用於處理 Pinecone 向量資料庫
from langchain_openai import OpenAIEmbeddings  # 用於處理 OpenAI 嵌入模型
from langchain.chat_models import init_chat_model  # 用於初始化聊天模型
from langchain.tools import tool  # 用於處理工具
from langchain.agents import create_agent  # 用於創建代理
from langchain.messages import ToolMessage  # 用於處理工具消息

# 初始化 OpenAI 嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 初始化 Pinecone 向量資料庫
vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)

# 初始化聊天模型
chat_model = init_chat_model(model="gpt-5-nano", model_provider="openai")


# 擷取文件工具
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """擷取相關文件，協助回答使用者關於 LangChain 的問題。"""
    # 擷取最相關的 4 份文件
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # 將文件整理成提供給模型的文字
    serialized = "\n\n".join(
        (f"來源：{doc.metadata.get('source', '未知來源')}\n\n內容：{doc.page_content}")
        for doc in retrieved_docs
    )

    # 同時回傳整理後的文字和原始文件
    # 因為上面標了 response_format="content_and_artifact"，這個 2-tuple 會依序拆進 ToolMessage：
    #   serialized     → ToolMessage.content   （送回給 LLM 閱讀，會計入 token）
    #   retrieved_docs → ToolMessage.artifact  （LLM 看不到，留給程式取用來源資訊）
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    使用擷取到的文件執行 RAG 流程，回答使用者的問題。

    參數：
        query：使用者的問題

    回傳：
        包含以下內容的字典：
            - answer：模型產生的答案
            - context：擷取到的文件清單
    """

    # 建立可使用文件檢索工具的代理
    system_prompt = (
        "你是一位樂於助人的 AI 助理，負責回答關於 LangChain 文件的問題。"
        "你可以使用工具擷取相關文件。"
        "回答問題前，請先使用工具尋找相關資訊。"
        "回答時務必引用所使用的來源。"
        "如果擷取到的文件中找不到答案，請明確說明。"
    )

    # 創建代理：加入文件檢索工具和系統提示
    agent = create_agent(chat_model, tools=[retrieve_context], system_prompt=system_prompt)

    # 建立使用者訊息清單　messages = [HumanMessage(content=query)]
    messages = [{"role": "user", "content": query}]

    # 呼叫代理
    response = agent.invoke({"messages": messages})

    # 從最後一則 AI 訊息中取出答案
    answer = response["messages"][-1].content

    # 從 ToolMessage 的 artifact 中取出作為上下文的文件
    context_docs = []
    for message in response["messages"]:  # 每一則訊息都看一遍
        # 確認這是否為包含 artifact 的 ToolMessage
        if (isinstance(message, ToolMessage)  # 如果它是工具訊息
                and hasattr(message, "artifact")):  # 而且有 artifact 屬性
            # artifact 預期包含 Document 物件清單
            if isinstance(message.artifact, list):  # 而且 artifact 是個清單
                context_docs.extend(message.artifact)  # 把裡面的東西全加進來

    return {
        "answer": answer,  # 從 messages[-1].content 來
        "context": context_docs  # 從 artifact 蒐集來
    }


if __name__ == "__main__":
    result = run_llm(query="什麼是 Deep Agents？")
    print(result)

# response 是 agent.invoke(...) 回傳的狀態字典；messages 依序記錄對話與工具呼叫結果。
# 以下內容為結構示意，實際訊息和欄位值會依執行結果而不同。
# response = {
#     "messages": [
#         HumanMessage(
#             content="LangGraph 是什麼？",  # 使用者提出的問題
#             id="8f3a2b10-..."  # 這則訊息的識別碼
#         ),
#         AIMessage(
#             content="",  # AI 這一步先要求呼叫工具，尚未產生回答文字
#             tool_calls=[{
#                 "name": "retrieve_context",  # 要呼叫的工具名稱
#                 "args": {"query": "LangGraph"},  # 傳給工具的參數
#                 "id": "call_abc123",  # 此次工具呼叫的識別碼
#                 "type": "tool_call"
#             }],
#             response_metadata={
#                 "model_name": "gpt-5-nano",
#                 "finish_reason": "tool_calls"  # AI 要求執行工具
#             },
#             usage_metadata={"input_tokens": 312, "output_tokens": 24, "total_tokens": 336},
#             id="run-2c9f..."
#         ),
#         ToolMessage(
#             content="來源：https://docs.langchain.com/...\n\n內容：LangGraph is a low-level...",  # 提供給 AI 閱讀的文字結果
#             name="retrieve_context",  # 回傳結果的工具名稱
#             tool_call_id="call_abc123",  # 對應上方 AIMessage.tool_calls 的 id
#             artifact=[  # 工具另外回傳的原始文件；run_llm 從這裡收集 context_docs
#                 Document(id="3f8a...", page_content="...", metadata={"source": "..."}),
#                 Document(id="7c1d...", page_content="...", metadata={"source": "..."}),
#                 Document(id="b4e9...", page_content="...", metadata={"source": "..."}),
#                 Document(id="e2f6...", page_content="...", metadata={"source": "..."}),
#             ],
#             id="9e4d1a..."
#         ),
#         AIMessage(
#             content="LangGraph 是一個低階的編排框架，用於建構有狀態的 agent...（來源：docs.langchain.com）",  # AI 讀取工具結果後產生的最終答案
#             response_metadata={
#                 "model_name": "gpt-5-nano",
#                 "finish_reason": "stop"  # 回答完成，這一步沒有再要求呼叫工具
#             },
#             usage_metadata={"input_tokens": 2841, "output_tokens": 186, "total_tokens": 3027},
#             id="run-5a7b..."
#         ),
#     ]
# }

# run_llm 從 ToolMessage.artifact 收集文件後，回傳的 context_docs 大致如下：
# context_docs = [
#     Document(
#         id="3f8a2c14-9b7e-4d21-a5f3-1e8c9d0b2a46",  # 文件識別碼
#         page_content="LangGraph is a low-level orchestration framework for building...",  # 文件片段文字
#         metadata={"source": "https://docs.langchain.com/oss/python/langgraph/"}  # 文件來源等中繼資料
#     ),
#     Document(
#         id="7c1d5e82-...",
#         page_content="State management in LangGraph is handled through...",
#         metadata={"source": "https://docs.langchain.com/oss/python/langgraph/persistence"}
#     ),
#     Document(
#         id="b4e9a037-...",
#         page_content="Nodes and edges define the control flow...",
#         metadata={"source": "https://docs.langchain.com/oss/python/langgraph/overview"}
#     ),
#     Document(
#         id="e2f6c8d1-...",
#         page_content="Checkpointing allows agents to resume...",
#         metadata={"source": "https://docs.langchain.com/oss/python/langgraph/add-memory"}
#     ),
# ]
