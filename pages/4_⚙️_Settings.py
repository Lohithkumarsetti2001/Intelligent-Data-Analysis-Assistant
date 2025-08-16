
import streamlit as st, os
st.set_page_config(page_title="Settings")
st.title("⚙️ Settings")
st.write("Configure models via environment variables in `.env`.")
st.code("""
HF_API_TOKEN=...
OPENAI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
""")
