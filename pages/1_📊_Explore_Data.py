
import streamlit as st, pandas as pd
from analysis.auto_analyzer import detect_types
st.set_page_config(page_title="Explore Data")
st.title("📊 Explore Data")
df = st.session_state.get("df")
if df is None:
    st.info("Load data on Home page.")
    st.stop()
st.dataframe(df, use_container_width=True)
st.json(detect_types(df))
