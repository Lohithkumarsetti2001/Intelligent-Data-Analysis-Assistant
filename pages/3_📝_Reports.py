
import streamlit as st
from utils.pdf_report import build_report
st.set_page_config(page_title="Reports")
st.title("📝 Reports")
insights = st.session_state.get("latest_insights", ["No insights yet."])
title = st.text_input("Title", "Data Analysis Report")
if st.button("Generate PDF"):
    build_report("analysis_report.pdf", title, insights, [])
    with open("analysis_report.pdf","rb") as f:
        st.download_button("Download PDF", f, file_name="analysis_report.pdf")
