
import streamlit as st
from ai_providers.router import multi_ai_generate
st.set_page_config(page_title="NL Queries")
st.title("🧠 Natural Language Queries")
q = st.text_area("Ask a question about your data or analysis")
if st.button("Ask") and q:
    sys = "Helpful, concise data assistant."
    resp = multi_ai_generate(prompt=q, system=sys)
    st.write(f"*LLM ({resp.provider})*: {resp.text}")
