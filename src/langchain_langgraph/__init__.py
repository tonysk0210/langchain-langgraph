import os  # 使用 os 模組來獲取環境變數

from dotenv import load_dotenv

load_dotenv()  # 載入環境變數 from .env file


def main() -> None:
    print("Hello from langchain-langgraph!")
    print(os.getenv("OPENAI_API_KEY"))


main()
