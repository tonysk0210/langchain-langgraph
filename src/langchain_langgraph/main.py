import os
from dotenv import load_dotenv

load_dotenv()  # 載入環境變數 from .env file

from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader  # 匯入 UnstructuredLoader 類別，用於處理非結構化資料
from langchain_text_splitters import CharacterTextSplitter  # 匯入 CharacterTextSplitter 類別，用於將文字分割成較小的片段
from langchain_pinecone import Pinecone  # 匯入 Pinecone 類別，用於與 Pinecone 向量資料庫互動

if __name__ == "__main__":
    print("Hello from langchain-langgraph!")
    print(os.environ["PINECONE_API_KEY"])  # 輸出環境變數 PINECONE_API_KEY 的值

    blog_path = Path(__file__).resolve().parents[2] / "blog.txt"
    loader = UnstructuredLoader(blog_path, chunking_strategy="basic", max_characters=1000000)  # 初始化 UnstructuredLoader，指定要載入的檔案路徑、分割策略和最大字元數
    document = loader.load()  # 載入檔案內容，返回一個文件物件

    print("正在將文件切分為多個片段...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # 初始化 CharacterTextSplitter，指定分割的片段大小和重疊字元數
    chunks = text_splitter.split_documents(document)  # 將文件分割成多個片段
    print(f"切分後的片段數量：{len(chunks)}")  # 輸出分割後的片段數量

    embeddings = OpenAIEmbeddings()  # 初始化 OpenAIEmbeddings，用於生成文字的向量表示

    print("正在將片段寫入 Pinecone...")
    pineconeVectorStore = Pinecone.from_documents(chunks, embeddings, index_name=os.environ["INDEX_NAME"])  # 將分割後的片段和向量表示存入 Pinecone 向量資料庫，指定索引名稱
    print("已完成將片段寫入 Pinecone")
