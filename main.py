# import os
# from dotenv import load_dotenv

# uv add streamlit-chat 
from typing import Any, List, Dict

import streamlit as st

from backend.core import run_llm

# load_dotenv()

def _format_source2(context_docs: List[Any]) -> List[str]:
    """
    INPUT
    [
        Doc(metadata={"source}: "wiki.pdf"}),
        Doc(metadata={"source}: "docs.txt"}),
    ]

    OUTPUT
    ["wiki.pdf", "docs.txt"]

    str( (meta.get("source") or "Unknown")) 
    for doc in context_docs 
    if (meta := {getattr(doc, "metadata", None)} or {}) is not None    
    """
    return [
        str(meta.get("source", "Unknown"))
        for doc in context_docs
        if (meta := getattr(doc, "metadata", {}))
     ]

def _format_source(context_docs: List[Any]) -> List[str]:
    sources = []
    for doc in context_docs:
        meta = getattr(doc, "metadata", {})
        sources.append(str(meta.get("source", "Unknown")))
    return sources


if __name__ == "__main__":
    """
    command line:
    uv run streamlit run main.py
    """
    print("Hello from streamlit.")

    st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
    st.title("LangChain Documentation Helper")

    with st.sidebar:
        st.subheader("Sessioon")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.pop("messages", None)
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me anything about LangChain docs. I'll retrieve relevant contxt and cite sourcer.",
                "sources": []  # ["www.langchain.com", "www.anthropic.com"],
            }

        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

    # prompt container
    prompt = st.chat_input("Ask a question about LangChain.")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving docs and generating answer .... wait ..."):
                    result: Dict[str, Any] = run_llm(prompt)
                    answer = str(result.get("answer", "nothing???")).strip() or "NO answer was returned."
                    sources = _format_source(result.get("context", []))

                st.markdown(answer)

            except Exception as e:
                st.error("Failed to generate a responde")
                st.exception(e)
