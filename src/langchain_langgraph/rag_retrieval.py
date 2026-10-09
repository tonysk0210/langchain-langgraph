import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate  # 匯入 ChatPromptTemplate 類別，用於建立聊天提示模板
from langchain_core.messages import HumanMessage  # 匯入 HumanMessage 類別，用於表示人類訊息
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 匯入 ChatOpenAI 和 OpenAIEmbeddings 類別，用於與 OpenAI API 互動
from langchain_pinecone import PineconeVectorStore  # 匯入 PineconeVectorStore 類別，用於與 Pinecone 向量資料庫互動

from langchain_core.runnables import RunnablePassthrough  # 匯入 RunnablePassthrough 類別，用於將輸入直接傳給輸出
from operator import itemgetter  # 匯入 itemgetter 函數，用於從字典中提取值
from langchain_core.output_parsers import StrOutputParser  # 匯入 StrOutputParser 類別，用於將輸出解析為字串

print("正在初始化元件...")

embeddings = OpenAIEmbeddings()  # 初始化 OpenAIEmbeddings，用於生成文字的向量表示
llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0)  # 初始化 ChatOpenAI，指定使用的模型名稱和溫度參數
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)  # 初始化 PineconeVectorStore，指定索引名稱和向量表示
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 將 PineconeVectorStore 轉換為檢索器，指定搜尋參數 k=3，表示每次檢索返回 3 個最相似的片段

# 建立提示模板，將檢索內容與使用者問題填入後提供給模型。
prompt_template = ChatPromptTemplate.from_template(
    """請僅根據以下內容回答問題：

{context}

問題：{question}

請提供詳細回答："""
)


def format_docs(docs):
    """將檢索到的文件整理成單一字串。"""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# 實作 1：不使用 LCEL（以簡單函式手動串接）
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    不使用 LCEL 的簡單檢索鏈。
    手動檢索文件、整理內容，並產生回答。

    限制：
    - 需要手動逐步執行
    - 沒有內建串流支援
    - 需要額外撰寫程式才能支援非同步
    - 較難與其他鏈組合
    - 程式碼較冗長，也較容易出錯
    """
    # 步驟 1：檢索相關文件
    docs = retriever.invoke(query)
    # 步驟 2：將文件整理成內容字串
    context = format_docs(docs)
    # 步驟 3：將檢索內容與問題填入提示模板
    messages = prompt_template.format_messages(context=context, question=query)
    # 步驟 4：將格式化後的訊息傳給 LLM
    response = llm.invoke(messages)
    # 步驟 5：回傳回答內容
    return response.content


# ============================================================================
# 實作 2：使用 LCEL（LangChain Expression Language，LangChain 表達式語言）——較佳做法
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    使用 LCEL（LangChain Expression Language）建立檢索鏈。
    回傳可透過 {"question": "..."} 傳入問題來執行的鏈。

    相較於不使用 LCEL 的做法，優點如下：
    - 宣告式且容易組合：可用管線運算子（|）串接操作
    - 內建串流：可直接使用 chain.stream()
    - 內建非同步：可使用 chain.ainvoke() 和 chain.astream()
    - 批次處理：可用 chain.batch() 一次處理多筆輸入
    - 型別安全：與 LangChain 的型別系統整合較好
    - 程式碼較少：內容更精簡、易讀
    - 可重複使用：可儲存、分享，也可與其他鏈組合
    - 易於除錯：LangChain 提供較完善的可觀測性工具
    """
    retrieval_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | retriever | format_docs
            )
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return retrieval_chain


# 這是一個測試範例，展示如何使用上述定義的函式。
if __name__ == "__main__":
    query = "Pinecone 在機器學習中是什麼？"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("實作 0：直接呼叫 LLM（不使用 RAG）")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("實作 1：不使用 LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("實作 2：使用 LCEL（較佳做法）")
    print("=" * 70)
    print("LCEL 的優點：")
    print("- 程式碼較精簡，且採用宣告式寫法")
    print("- 內建串流功能：chain.stream()")
    print("- 內建非同步功能：chain.ainvoke()")
    print("- 容易與其他鏈組合")
    print("- 較適合用於正式環境")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer：")
    print(result_with_lcel)
