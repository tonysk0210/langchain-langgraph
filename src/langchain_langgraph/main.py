# import os  # 使用 os 模組來獲取環境變數

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()  # 載入環境變數 from .env file


def main() -> None:
    print("Hello from langchain-langgraph!")
    # print(os.getenv("OPENAI_API_KEY"))

    information = """
    馬斯克出生於南非比勒陀利亞的富裕家庭，1989年隨家人移民加拿大；由於母親出生於加拿大，他擁有加拿大國籍。 1997年，他從賓州大學獲得學士學位，之後前往加州發展商業。 1995年，馬斯克共同創辦了網路軟體公司Zip2。 1999年Zip2出售後，他又共同創辦了電子商務支付系統X.com 。 2000年3月， X.com與Confinity合併，成立了PayPal，後者於2002年被eBay收購。同年，馬斯克也加入了美國國籍。
    2002年，馬斯克創立了太空科技公司SpaceX，並擔任執行長兼首席工程師；該公司此後在可重複使用火箭和商業航太領域引領創新。 2004年，馬斯克以早期投資者身分加入特斯拉，並於2008年成為其執行長兼產品架構師；特斯拉此後成為電動車領域的領導者。 2015年，馬斯克共同創立了OpenAI，旨在推進人工智慧（AI）研究，但後來離開了該公司。他對特斯拉在2020年代人工智慧熱潮中的發展方向和領導層日益不滿，這促使他創立了xAI，該公司於2026年成為SpaceX的子公司。 2022年，他收購了社交網路服務公司Twitter ，並對其進行了重大改革，包括在2023年將其更名為X。他的其他企業包括他於2016年共同創立的神經科技公司Neuralink ，以及他於2017年創立的隧道公司The Boring Company。 2025年11月，特斯拉批准了一項價值1兆美元的薪酬方案，如果特斯拉達到某些里程碑（例如市值達到8.5兆美元），馬斯克將在10年內分期領取這筆款項。 2026年6月SpaceX首次公開募股後，馬斯克成為首位身價兆美元的美國富豪，但SpaceX股價暴跌，導致他隔月便跌破了這一門檻。
    """
    prompt_template = """
    請用以下資料來總結馬斯克的生平：
    1. 簡短的摘要
    2. 兩個有趣的關鍵點
    {information}
    """
    # 初始化 LLM
    openai_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    ollama_llm = ChatOllama(model="gemma3:4b", temperature=0)

    # 1. 將 prompt template 轉換為 prompt
    prompt = PromptTemplate.from_template(
        prompt_template
    )  
    # 2. 將 prompt 和 llm 串接起來
    chain = prompt | openai_llm  
    # 3. 執行 chain
    response = chain.invoke(input={"information": information})  
    # 4. 印出 response
    print(response.content)  


main()
