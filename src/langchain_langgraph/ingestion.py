import asyncio
import os  # 用於處理作業系統相關的任務
import ssl  # 用於處理 SSL 連線
from typing import List  # 用於型別提示

from dotenv import load_dotenv  # 用於載入環境變數

load_dotenv()  # 載入環境變數
import certifi  # 用於處理 SSL 證書
from langchain_core.documents import Document  # 用於表示文件
from langchain_openai import OpenAIEmbeddings  # 用於將文字轉換為向量
from langchain_pinecone import PineconeVectorStore  # 用於將向量存儲在 Pinecone 中
from langchain_tavily import TavilyCrawl  # 用於從 Tavily 提取資料
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 用於將文字分割成較小的片段

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)  # 用於輸出日誌

# 設定 SSL 上下文以使用 certifi 證書
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# 初始化 OpenAI embedding 模型，用於將文字轉換為向量
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,  # 每次最多把 50 段文字送給 OpenAI 轉成向量
    retry_min_seconds=10,
)

# 初始化 Pinecone 向量存儲，用於存儲文件的向量表示
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)

# 初始化 Tavily 爬蟲，用於從 Tavily 網站爬取資料
tavily_crawl = TavilyCrawl()


# 以非同步方式分批處理文件
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """以非同步方式分批處理文件。"""
    log_header("向量儲存階段")
    log_info(
        f"📚 向量資料庫索引：準備新增 {len(documents)} 份文件",
        Colors.DARKCYAN,
    )

    # 將文件分批
    batches = [
        documents[i: i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 向量資料庫索引：已分成 {len(batches)} 批，每批最多 {batch_size} 份文件"
    )

    # 定義「單一批次」的寫入動作；失敗只影響這一批，回傳 True／False ============================================
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)  # 非同步：先呼叫 OpenAI 把文字轉成向量，再寫入 Pinecone
            log_success(
                f"向量資料庫寫入：第 {batch_num}/{len(batches)} 批已成功新增（{len(batch)} 份文件）"
            )
        except Exception as e:
            log_error(f"向量資料庫寫入：第 {batch_num} 批新增失敗 - {e}")
            return False
        return True

    # ====================================================================================================

    # 為每一批建立協程（此時尚未執行），再交給 gather 一次全部併發送出
    tasks = [add_batch(batch, i + 1)
             for i, batch in enumerate(batches)]
    # 并發執行所有協程，等待所有任務完成
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 計算成功處理的批次數
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"向量資料庫寫入：所有批次皆已成功處理！（{successful}/{len(batches)}）"
        )
    else:
        log_warning(
            f"向量資料庫寫入：已成功處理 {successful}/{len(batches)} 批"
        )


async def main():
    """負責協調整個流程的非同步主函式。"""
    log_header("文件資料匯入流程")
    log_info(
        "🔍 TavilyCrawl：開始爬取 LangChain 文件：https://python.langchain.com/",
        Colors.PURPLE,
    )

    # 爬取 LangChain 文件，default limit = 50 連結
    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://python.langchain.com/",
            "extract_depth": "advanced",
            "instructions": "Documentation relevant to ai agents",  # 這段文字會送給 Tavily 的 Crawl API，作為爬取方向的自然語言提示
            "max_depth": 3,
        }
    )

    # 檢查爬取是否成功
    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl：爬取失敗：{tavily_crawl_results['error']}")
        return
    else:
        log_success(
            f"TavilyCrawl：成功爬取文件網站的 {len(tavily_crawl_results.get('results', []))} 個頁面"
        )

    # 將爬取到的文件內容轉換為 Document 類型
    all_docs = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        log_info(
            f"TavilyCrawl：成功爬取文件網站頁面：{tavily_crawl_result_item['url']}"
        )

        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],  # 字串，表示文件內容
                metadata={"source": tavily_crawl_result_item["url"]},  # 字典，表示文件來源
            )
        )

    # 計算文件內容長度
    lengths = [len(d.page_content) for d in all_docs]
    log_info(f"內容長度：最短 {min(lengths)}／最長 {max(lengths)}／平均 {sum(lengths) // len(lengths)}")

    # 將文件切分為文字片段
    log_header("文件切片階段")
    log_info(
        f"✂️ 文字切分器：以每段 4000 字元、重疊 200 字元的設定處理 {len(all_docs)} 份文件",
        Colors.YELLOW,
    )
    # 初始化文字切分器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)  # 切分文字片段 4000 字元，重疊 200 字元
    # 將文件切分為文字片段
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"文字切分器：已將 {len(all_docs)} 份文件切分為 {len(splitted_docs)} 個片段"
    )

    # 以非同步方式處理片段
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("匯入流程完成")
    log_success("🎉 文件資料匯入流程已成功完成！")
    log_info("📊 執行摘要：", Colors.BOLD)
    log_info(f"   • 已爬取頁面：{len(tavily_crawl_results.get('results', []))}")
    log_info(f"   • 已擷取文件：{len(all_docs)}")
    log_info(f"   • 已建立文字片段：{len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
