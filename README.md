
# Intelligent Data Analysis Assistant

A Streamlit-based, multi-AI, end-to-end data analysis assistant.

## Features (maps to your rubric)
- Professional Streamlit UI with custom CSS
- Multi‑AI integration: Hugging Face Inference API (optional) + Ollama local (default) with fallback
- Pandas/Numpy processing with caching
- Matplotlib & Seaborn visualizations
- NLTK NLP: tokenization, stopword removal, VADER sentiment
- Intelligent query processing: plain‑English → actions (rules + LLM fallback)
- Automated PDF reporting (ReportLab) with insights and charts
- Session management: history, preferences
- Export: CSV of processed data, PNGs of charts, PDF report

## Quickstart
```bash
# 1) Create & activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) First‑time NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# 4) (Optional) Copy .env.example to .env and add keys.
#    If you have Ollama running, that's enough.

# 5) Run app
streamlit run app.py
```

If `streamlit` is not recognized on Windows, make sure your virtual env is activated and try:
`.\.venv\Scripts\python -m streamlit run app.py`

