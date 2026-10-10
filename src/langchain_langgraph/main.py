from typing import Any, Dict, List  # 用於型別提示
import streamlit as st  # 用於建立Web應用介面
from core import run_llm  # 用於運行 LLM


# 對每一份文件，取出它 metadata 裡的 source；取不到就寫 "Unknown"。回傳一個字串清單。
def _format_sources(context_docs: List[Any]) -> List[str]:
    sources = []
    for doc in context_docs or []:
        meta = getattr(doc, "metadata", None) or {}
        sources.append(str(meta.get("source") or "Unknown"))
    return sources


# 設定網頁標題和布局
st.set_page_config(page_title="LangChain RAG 文件助理", layout="centered")
st.markdown(
    "<h1 style='text-align: center;'>LangChain RAG 文件助理</h1>",
    unsafe_allow_html=True,
)

# 建立邊欄
with st.sidebar:
    st.subheader("Session")
    if st.button("清除對話", use_container_width=True):
        st.session_state.pop("messages", None)  # 清除對話紀錄 只刪 messages 一個鍵：Server 記憶體；session_state 是一個字典
        st.rerun()  # 重新運行應用程式

# 如果 session_state 沒有 messages 這個鍵，就建立一個預設的對話紀錄。預設的對話紀錄只有一條訊息，內容是歡迎訊息，且沒有資料來源。
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "歡迎詢問任何關於 LangChain 文件的問題。我會擷取相關內容並引用資料來源。",
            "sources": [],
        }
    ]

# 顯示對話紀錄
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # 氣泡 設定角色
        st.markdown(msg["content"])  # 訊息內容
        if msg.get("sources"):
            with st.expander("資料來源"):  # 氣泡「裡面」的摺疊區
                for s in msg["sources"]:
                    st.markdown(f"- {s}")  # 摺疊區「裡面」的清單

# 使用 chat_input 建立輸入框，提示使用者輸入關於 LangChain 的問題
prompt = st.chat_input("請輸入關於 LangChain 的問題...")

# 如果使用者有輸入問題，就將問題加入對話紀錄，並顯示問題
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})  # 將問題加入對話紀錄，並顯示問題
    with st.chat_message("user"):  # 氣泡 設定角色
        st.markdown(prompt)

    with st.chat_message("assistant"):  # 氣泡 設定角色
        # 輸出答案
        try:
            with st.spinner("正在擷取文件並產生回答..."):
                # 呼叫 run_llm 函數，傳入問題，回傳結果
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "（沒有取得回答。）"
                sources = _format_sources(result.get("context", []))  # 取得資料來源

            st.markdown(answer)  # 顯示答案
            if sources:
                with st.expander("資料來源"):  # 氣泡「裡面」的摺疊區
                    for s in sources:  # 摺疊區「裡面」的清單
                        st.markdown(f"- {s}")  # 顯示資料來源

            # 將答案加入對話紀錄
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("產生回答時發生錯誤。")
            st.exception(e)

# uv run streamlit run src\langchain_langgraph\main.py
