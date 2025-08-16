# app.py
import os
import io
import json
import time
import requests
import traceback
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_lottie import st_lottie

# Optional imports (used if available)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:
    nltk = None

# Hugging Face / transformers optional
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Optional user-provided PDF builder from your repo
try:
    from utils.pdf_report import build_report
    HAVE_BUILD_REPORT = True
except Exception:
    HAVE_BUILD_REPORT = False

# Load environment variables
load_dotenv()

# ---------------- Configuration ----------------
st.set_page_config(page_title="Intelligent Data Analysis Assistant",
                    layout="wide", page_icon="🤖")
CHART_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#64B5CD"]
ASSETS_CSS = "assets/styles.css"

# ---------------- Helpers & Caching ----------------
def local_css(file_name: str = ASSETS_CSS):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # fallback CSS (minimal)
        st.markdown(
            """
            <style>
            :root{--accent1:#4C72B0;--accent2:#64B5CD}
            .hero{padding:18px;border-radius:12px;margin-bottom:12px}
            .card{padding:12px;border-radius:10px;margin-bottom:10px}
            .answer-box{animation:fadeIn 0.9s ease-in;padding:12px;border-radius:8px}
            @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
            .centered-loader{display:flex;align-items:center;justify-content:center;margin:12px 0}
            .rotating-loader{width:42px;height:42px;border:6px solid rgba(0,0,0,0.06);border-top-color:var(--accent1);border-radius:50%;animation:spin 1s linear infinite}
            .stopped-loader{width:42px;height:42px;border-radius:50%;background: linear-gradient(135deg,var(--accent1),var(--accent2));opacity:0.95}
            @keyframes spin{to{transform:rotate(360deg)}}
            </style>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def read_csv_bytes(uploaded_file) -> pd.DataFrame:
    """Read uploaded file reliably and cache result."""
    return pd.read_csv(uploaded_file)


def try_build_text_pdf(path: str, title: str, insights: list, charts: list):
    """Fallback PDF generator: writes a simple text-based report."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(title + "\n\n")
            f.write("Automated insights:\n")
            for i in insights:
                f.write("- " + str(i) + "\n")
            f.write("\nNote: For a full PDF builder, provide utils.pdf_report.build_report.\n")
        return True
    except Exception:
        return False


# ---------------- Hugging Face / LLM Helpers ----------------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip() or None
HF_REMOTE_URL = "https://api-inference.huggingface.co/models/"

def hf_remote_request(model: str, inputs: str, params: dict = None) -> Optional[str]:
    """
    Use Hugging Face Inference API if token present.
    Returns text or None on failure.
    """
    if not HF_API_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        url = HF_REMOTE_URL + model
        payload = {"inputs": inputs}
        if params:
            payload["parameters"] = params
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # many models return string, some return list of dicts
        if isinstance(data, list):
            # common summarization / generation returns [{"generated_text": "..."}]
            candidate = data[0]
            if isinstance(candidate, dict):
                return candidate.get("generated_text") or candidate.get("summary_text") or str(candidate)
            return str(data)
        if isinstance(data, dict):
            # if returned dict with 'generated_text'
            return data.get("generated_text") or data.get("summary_text") or str(data)
        return str(data)
    except Exception:
        return None


# Try to create local transformers pipelines if available (lazy)
LOCAL_PIPELINES = {}
def get_transformer_pipeline(task: str, model_name: str = None):
    """
    Return a transformers pipeline for 'summarization', 'question-answering', or 'sentiment-analysis' if possible.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None
    key = f"{task}:{model_name}"
    if key in LOCAL_PIPELINES:
        return LOCAL_PIPELINES[key]
    try:
        if task == "summarization":
            mdl = model_name or "facebook/bart-large-cnn"
            pipe = pipeline("summarization", model=mdl, tokenizer=mdl)
        elif task == "question-answering":
            mdl = model_name or "distilbert-base-cased-distilled-squad"
            pipe = pipeline("question-answering", model=mdl, tokenizer=mdl)
        elif task == "sentiment-analysis":
            mdl = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
            pipe = pipeline("sentiment-analysis", model=mdl, tokenizer=mdl)
        else:
            return None
        LOCAL_PIPELINES[key] = pipe
        return pipe
    except Exception:
        return None


def llm_answer(prompt: str, mode: str = "summarize") -> dict:
    """
    Try: local transformer pipeline -> HF remote -> Ollama endpoints -> fallback message.
    mode: 'summarize' | 'qa' | 'sentiment' | 'generate'
    Returns dict: {"text": str, "provider": str}
    """
    # 1) local transformers pipeline
    if mode == "summarize":
        p = get_transformer_pipeline("summarization")
        if p:
            try:
                out = p(prompt, max_length=200, min_length=20, do_sample=False)
                text = out[0]["summary_text"] if isinstance(out, list) and "summary_text" in out[0] else out[0].get("summary_text") or out[0].get("generated_text") or str(out)
                return {"text": str(text), "provider": "transformers_local"}
            except Exception:
                pass
        # remote HF
        out = hf_remote_request("facebook/bart-large-cnn", prompt, params={"max_length": 200, "min_length": 20})
        if out:
            return {"text": out, "provider": "huggingface_remote"}

    if mode == "qa":
        p = get_transformer_pipeline("question-answering")
        if p:
            try:
                # for QA we expect prompt like "context: ... \n question: ..."
                parts = prompt.split("\nquestion:")
                if len(parts) == 2:
                    context = parts[0].replace("context:", "").strip()
                    question = parts[1].strip()
                else:
                    # fallback: treat last sentence as question
                    question = prompt.strip().split("?")[-1]
                    context = prompt
                out = p(question=question, context=context)
                return {"text": out.get("answer", str(out)), "provider": "transformers_local"}
            except Exception:
                pass
        out = hf_remote_request("deepset/roberta-base-squad2", prompt)
        if out:
            return {"text": out, "provider": "huggingface_remote"}

    if mode == "sentiment":
        p = get_transformer_pipeline("sentiment-analysis")
        if p:
            try:
                out = p(prompt)
                # out often list like [{'label': 'POSITIVE', 'score': 0.999...}]
                if isinstance(out, list) and isinstance(out[0], dict):
                    label = out[0].get("label")
                    score = out[0].get("score")
                    return {"text": f"{label} (score={score:.3f})", "provider": "transformers_local"}
                return {"text": str(out), "provider": "transformers_local"}
            except Exception:
                pass
        out = hf_remote_request("distilbert-base-uncased-finetuned-sst-2-english", prompt)
        if out:
            return {"text": out, "provider": "huggingface_remote"}

    # 2) Ollama endpoints (best-effort fallback)
    ollama_endpoints = [
        "http://127.0.0.1:11434/api/generate",
        "http://127.0.0.1:11434/v1/generate"
    ]
    payload = {"model": os.environ.get("OLLAMA_MODEL", "gemma3:1b"), "prompt": prompt, "max_tokens": 300}
    for url in ollama_endpoints:
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.ok:
                # server may stream chunks; join responses
                text = ""
                for line in (r.text or "").splitlines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        text += d.get("response", "") or d.get("completion", "") or d.get("content", "") or ""
                    except Exception:
                        # not json -> append raw
                        text += line
                if not text:
                    # try parse whole
                    try:
                        j = r.json()
                        if isinstance(j, dict):
                            text = j.get("response") or j.get("completion") or j.get("message", {}).get("content") or str(j)
                        elif isinstance(j, list):
                            text = " ".join([str(x) for x in j])
                    except Exception:
                        text = r.text
                return {"text": text.strip(), "provider": f"ollama ({url})"}
        except Exception:
            continue

    return {"text": "Could not reach any LLM. Please check transformers/HF token/Ollama.", "provider": "fallback"}


def dynamic_prompt_builder(query: str, df: pd.DataFrame) -> Optional[str]:
    """
    Analyzes a user query and attempts to extract relevant data from the DataFrame
    to build a more effective prompt for the LLM.
    """
    # 1. Identify a potential column name in the query
    lower_query = query.lower()
    possible_cols = [col for col in df.columns if col.lower() in lower_query]

    # 2. If a column is found, extract some sample data.
    if possible_cols:
        col = possible_cols[0]
        # Get up to 100 random, non-null values from that column.
        sample_data = df[col].dropna().sample(min(100, len(df))).tolist()

        # 3. Build a detailed prompt with the sample data as context.
        # This gives the LLM the information it needs.
        data_string = ", ".join(map(str, sample_data))
        prompt = f"""
        Analyze the following data from a column named '{col}' and answer the user's question.

        Data from '{col}': {data_string}

        User's question: {query}
        """
        return prompt
    else:
        # If no column is identified, just return the original query.
        return query


def advanced_data_query(query: str, df: pd.DataFrame) -> Optional[str]:
    """
    Tries to perform a direct data lookup from the DataFrame first.
    If no specific lookup is found, it falls back to the dynamic prompt builder.
    """
    lower_query = query.lower()
    
    # Simple check for a lookup pattern, e.g., "name for id 1", "id 1 name"
    if "name" in lower_query and "id" in lower_query:
        # Attempt to extract the ID number
        try:
            # Simple regex to find a number after "id" or "passengerId"
            id_str = "".join(filter(str.isdigit, lower_query.split("id")[-1].strip()))
            passenger_id = int(id_str)
            
            # Find the row with the matching PassengerId (assuming it's a unique key)
            result = df[df["PassengerId"] == passenger_id]["Name"].values
            
            if len(result) > 0:
                name = result[0]
                return f"The name for PassengerId {passenger_id} is: {name}"
        except (ValueError, IndexError):
            # If parsing fails, fall through to the LLM
            pass
            
    # Fallback to the LLM for more general questions or if the lookup failed
    return dynamic_prompt_builder(query, df)


# ---------------- NLTK helpers ----------------
def ensure_nltk():
    if not nltk:
        return False
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except Exception:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except Exception:
        nltk.download("averaged_perceptron_tagger", quiet=True)
    return True


def text_preprocess_basic(text: str):
    """Tokenize, lowercase, remove stopwords, lemmatize (best effort)."""
    if not nltk:
        return {"tokens": [], "lemmas": [], "top_words": []}
    ensure_nltk()
    toks = nltk.word_tokenize(str(text))
    toks = [t.lower() for t in toks if t.isalpha()]
    sw = set(stopwords.words("english"))
    toks_nosw = [t for t in toks if t not in sw]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in toks_nosw]
    freq = pd.Series(lemmas).value_counts().head(10)
    return {"tokens": toks, "lemmas": lemmas, "top_words": freq.to_dict()}


# ---------------- App UI & Flow ----------------
local_css()
# Load Lottie animations (best-effort)
lottie_success = None
lottie_loading = None
lottie_wave = None
try:
    lottie_success = load_lottieurl = None  # placeholder to avoid unused code issues
except Exception:
    pass

# safe load lottie URLs
def _safe_load(url):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

lottie_success = _safe_load("https://assets1.lottiefiles.com/packages/lf20_jbrw3hcz.json")
lottie_loading = _safe_load("https://assets3.lottiefiles.com/packages/lf20_usmfx6bp.json")
lottie_wave = _safe_load("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "preferences" not in st.session_state:
    st.session_state.preferences = {"target_column": None}
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

# Sidebar - keep minimal until dataset loaded (you asked to move controls to main after load)
with st.sidebar:
    st.markdown("<h3>🤖 Controls</h3>", unsafe_allow_html=True)
    theme_choice = st.radio("Theme", ["Light", "Dark"], index=1 if st.session_state.theme == "dark" else 0)
    st.session_state.theme = theme_choice.lower()
    st.markdown("---")
    st.markdown("Need help? See the cards on the right after loading a dataset.")

# Main: uploader area (if not loaded)
if not st.session_state.file_loaded:
    st.markdown("<div class='hero'><h1>🤖 Intelligent Data Analysis Assistant</h1>"
                 "<p style='margin-top:6px'>Upload a CSV to begin or load a sample dataset</p></div>",
                 unsafe_allow_html=True)
    cols = st.columns([2, 1])
    with cols[0]:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = read_csv_bytes(uploaded)
                st.session_state.df = df
                st.session_state.file_loaded = True
                # refresh to show main UI: attempt to rerun
                try:
                    st.experimental_rerun()
                except Exception:
                    try:
                        st.rerun()
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
        if st.button("Load Sample Dataset"):
            sample_path = os.path.join("data", "sample.csv")
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                st.session_state.df = df
                st.session_state.file_loaded = True
                try:
                    st.experimental_rerun()
                except Exception:
                    try:
                        st.rerun()
                    except Exception:
                        pass
            else:
                st.error("Sample dataset not found at data/sample.csv")
    with cols[1]:
        st.markdown("<div class='card'><h4>Quick Tips</h4>"
                     "<ul><li>CSV with header row works best</li>"
                     "<li>Numeric columns are used for charts</li>"
                     "<li>Use the 'Speak' button for voice queries (Chrome/Edge)</li></ul></div>",
                     unsafe_allow_html=True)
    st.stop()

# If reached here, dataset is loaded
df: pd.DataFrame = st.session_state.df

# Top header (post-load)
theme = st.session_state.theme
card_bg = "#1c1f2a" if theme == "dark" else "#ffffff"
text_color = "#ffffff" if theme == "dark" else "#111111"

st.markdown(f"""
<div class='hero' style='background: linear-gradient(90deg, #0f111a, #18202a); color:{text_color};'>
  <div style='display:flex; gap:16px; align-items:center'>
    <div style='width:64px; height:64px; border-radius:12px; background:linear-gradient(135deg,{CHART_COLORS[0]},{CHART_COLORS[4]}); display:flex; align-items:center; justify-content:center; font-size:28px;'>🤖</div>
    <div>
      <h2 style='margin:0;color:{text_color};'>Intelligent Data Analysis Assistant</h2>
      <div style='color:rgba(255,255,255,0.85)'>Smart analytics • NLP queries (speech supported) • Multi-LLM fallback</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Layout: left (main) + right (controls & meta)
left, right = st.columns([3, 1])

# MAIN COLUMN
with left:
    st.markdown(f"<div class='card' style='background:{card_bg}; color:{text_color};'><h3>📋 Dataset Preview</h3></div>", unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True)

    # Quick summary
    st.markdown(f"<div class='card' style='background:{card_bg}; color:{text_color};'><h3>📊 Quick Summary & Types</h3></div>", unsafe_allow_html=True)
    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_pct": float(df.isna().mean().mean()) * 100,
        "numeric_cols": int(len(df.select_dtypes(include="number").columns)),
        "categorical_cols": int(len(df.select_dtypes(include="object").columns))
    }
    st.json(summary)
    types = {c: ("numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical") for c in df.columns}
    st.json(types)

    # Automated analysis + insights
    if st.button("Run Automatic Analysis", key="auto_analysis_main"):
        if lottie_loading:
            st_lottie(lottie_loading, height=120)
        insights = []
        if st.session_state.preferences.get("target_column"):
            t = st.session_state.preferences["target_column"]
            if t in df.columns and pd.api.types.is_numeric_dtype(df[t]):
                corr = df.corr().get(t, pd.Series()).dropna().sort_values(ascending=False).head(5)
                insights.append(f"Top correlations with {t}: {', '.join([f'{idx} ({val:.2f})' for idx,val in corr.items() if idx!=t])}")
            else:
                insights.append("Selected target is not numeric; choose numeric for correlation insights.")
        else:
            insights.append("No target selected — choose a target column to get correlation insights.")

        st.session_state["latest_insights"] = insights
        for i in insights:
            st.info(i)
        if lottie_success:
            st_lottie(lottie_success, height=90)

    # Charts
    st.markdown(f"<div class='card' style='background:{card_bg}; color:{text_color};'><h3>📈 Interactive Visualizations</h3></div>", unsafe_allow_html=True)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        hist_col = st.selectbox("Choose numeric column for histogram", num_cols, index=0)
        if hist_col:
            fig = px.histogram(df, x=hist_col, nbins=30, color_discrete_sequence=CHART_COLORS, opacity=0.9, template="plotly_white")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color), transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

        if len(num_cols) >= 2 and len(num_cols) <= 40:
            with st.expander("Show correlation heatmap"):
                corr = df[num_cols].corr()
                fig2 = px.imshow(corr, text_auto=True, color_continuous_scale=["#ffffff", CHART_COLORS[0]], zmin=-1, zmax=1)
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color))
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No numeric columns available for charts.")

    # NLP box: voice + text
    st.markdown(f"<div class='card' style='background:{card_bg}; color:{text_color};'><h3>🧠 Natural Language Queries (Text or Voice)</h3></div>", unsafe_allow_html=True)

    # Voice component: same approach as before. keep plain string to avoid brace formatting issues
    voice_html = """
    <div style="display:flex; gap:10px; align-items:center;">
      <button id="voiceBtn" style="padding:8px 12px; border-radius:8px; border:none; background:linear-gradient(135deg,#4C72B0,#64B5CD); color:white; font-weight:600; cursor:pointer;">
        🎙️ Speak
      </button>
      <div style="color: rgba(255,255,255,0.85); font-size:13px;">Click Speak, allow microphone, then speak. The page will reload with your transcript.</div>
    </div>
    <script>
    (function(){
      const btn = document.getElementById('voiceBtn');
      if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
          btn.innerText = '🎙️ Not supported';
          btn.style.opacity = 0.6;
          btn.onclick = function(){ alert('Speech recognition not supported in this browser. Use Chrome/Edge.'); }
      } else {
          const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
          const recognition = new SpeechRecognition();
          recognition.lang = 'en-US';
          recognition.interimResults = false;
          recognition.maxAlternatives = 1;
          recognition.onstart = function(){ btn.innerText = 'Listening...'; }
          recognition.onend = function(){ btn.innerText = '🎙️ Speak'; }
          recognition.onerror = function(e){ console.log('Speech error', e); alert('Speech recognition error: ' + (e && e.error ? e.error : e)); btn.innerText = '🎙️ Speak'; }
          recognition.onresult = function(event){
              const transcript = event.results[0][0].transcript;
              const encoded = encodeURIComponent(transcript);
              const base = window.location.pathname;
              window.location.href = base + '?voice=' + encoded;
          }
          btn.onclick = function(){ recognition.start(); }
      }
    })();
    </script>
    """
    components.html(voice_html, height=90)

    # Prefill from voice query (if present)
    voice_prefill = st.query_params.get("voice", [""])[0] if hasattr(st, "query_params") else ""
    nlp_default = voice_prefill or ""
    natural_q = st.text_area("Ask a question (or paste transcript)", value=nlp_default, height=100, placeholder="E.g. Summarize column X; Show top 3 correlations with column Y")
    query_mode = st.selectbox("Mode", ["Summarize dataset / column", "Question-Answer (context + question)", "Sentiment (text)"], index=0)

    run_q = st.button("Run Query", key="nlp_run")
    result_area = st.empty()
    loader = st.empty()

    if run_q and natural_q.strip():
        # show rotating loader
        loader.markdown("<div class='centered-loader'><div class='rotating-loader'></div></div>", unsafe_allow_html=True)
        if lottie_wave:
            st_lottie(lottie_wave, height=80, key=f"wave-{time.time()}")

        # prepare prompt and mode
        try:
            # Use the NEW advanced data query function here
            prompt_text = advanced_data_query(natural_q, df)
            
            # If the advanced query returned a direct answer, display it without the LLM
            if prompt_text and not prompt_text.startswith("Analyze the following data"):
                 out = {"text": prompt_text, "provider": "direct lookup"}
            else:
                # Otherwise, send the prepared prompt to the LLM
                if query_mode.startswith("Summarize"):
                    out = llm_answer(prompt_text, mode="summarize")
                elif query_mode.startswith("Question-Answer"):
                    # expect user to provide "context: ... \nquestion: ..."
                    out = llm_answer(prompt_text, mode="qa")
                else:
                    out = llm_answer(prompt_text, mode="sentiment")
        except Exception as e:
            out = {"text": f"LLM call failed: {e}\n{traceback.format_exc()}", "provider": "error"}

        # stop loader -> show stopped loader briefly
        loader.markdown("<div class='centered-loader'><div class='stopped-loader'></div></div>", unsafe_allow_html=True)
        time.sleep(0.25)
        loader.empty()

        result_area.markdown(f"""
        <div class="answer-box" style="background:{card_bg}; color:{text_color}; padding:12px; border-radius:10px;">
          <b>LLM ({out.get('provider','unknown')}):</b>
          <div style="margin-top:8px; white-space:pre-wrap;">{out.get('text', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.history.append({"q": natural_q, "mode": query_mode, "result": out.get("text", ""), "provider": out.get("provider","")})

    # NLTK text analysis quick card (if text columns exist)
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if text_cols:
        with st.expander("Run quick NLP on a text column (tokenize, top tokens)"):
            col = st.selectbox("Choose text column", ["(none)"] + text_cols, index=0)
            if col != "(none)":
                sample_text = " ".join(df[col].dropna().astype(str).head(200).tolist())
                processed = text_preprocess_basic(sample_text)
                st.write("Top tokens / lemmas:")
                st.json(processed.get("top_words", {}))
    else:
        st.info("No text/categorical columns detected for NLP quick analysis.")


# RIGHT / CONTROLS COLUMN
with right:
    st.markdown(f"<div class='card' style='background:{card_bg}; color:{text_color};'><h4>⚙️ Controls & Metadata</h4></div>", unsafe_allow_html=True)
    # target select but do not show 'None' text — show '— none —' nicely
    cols_for_target = ["— none —"] + df.columns.tolist()
    sel_idx = 0
    try:
        # remember previous choice if valid
        prev = st.session_state.preferences.get("target_column")
        if prev in df.columns:
            sel_idx = cols_for_target.index(prev)
    except Exception:
        sel_idx = 0
    sel = st.selectbox("Target column (optional)", cols_for_target, index=sel_idx)
    st.session_state.preferences["target_column"] = None if sel == "— none —" else sel

    st.markdown("---")
    if st.button("Refresh page"):
        try:
            st.rerun()
        except Exception:
            # fallback: set a query param to force reload with the new API
            if hasattr(st, "query_params"):
                st.query_params.clear()
            else:
                st.experimental_rerun()


    if st.button("Clear history"):
        st.session_state.history = []
        st.success("History cleared")

    st.markdown("---")
    st.write(f"History items: {len(st.session_state.history)}")
    st.markdown("Export / Reports")
    if st.button("Generate PDF report"):
        # attempt to use build_report if available
        insights = st.session_state.get("latest_insights", ["Run analysis first."])
        pdf_path = "analysis_report.pdf"
        try:
            if HAVE_BUILD_REPORT:
                build_report(pdf_path, "Data Analysis Report", insights, [])
                st.success("PDF generated - use the button below to download")
            else:
                ok = try_build_text_pdf(pdf_path, "Data Analysis Report", insights, [])
                if ok:
                    st.success("Text-based PDF (fallback) generated - use button below to download")
                else:
                    st.error("Could not generate PDF (missing helper).")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.error(traceback.format_exc())

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("Download report", f, file_name=pdf_path)

    st.caption("Tip: Use concise NLP queries for best results.")
