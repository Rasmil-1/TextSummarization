import streamlit as st
import streamlit.components.v1 as components
import nltk
import numpy as np
import re
import io
import plotly.graph_objects as go

from summarizer import summarize
from evaluate import get_random_article
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer as rouge_module
import textstat

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

st.set_page_config(
    page_title="NewsSnap · NLP Text Summarizer",
    layout="wide",
    page_icon="📰",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-page: #0d0d11;
    --bg-elevated: #121218;
    --bg-card: #16161d;
    --bg-input: #1c1c26;
    --border: #2a2a36;
    --border-strong: #3a3a48;
    --accent: #8a70ff;
    --accent-mid: #7a62e8;
    --accent-deep: #6a50df;
    --gradient-primary: linear-gradient(90deg, #8a70ff 0%, #6a50df 100%);
    --gradient-primary-hover: linear-gradient(90deg, #9a82ff 0%, #7a60ef 100%);
    --text-primary: #ffffff;
    --text-body: #e8e8f0;
    --text-secondary: #c8c8d4;
    --text-label: #9494a3;
    --radius: 15px;
    --shadow-card: 0 2px 12px rgba(0,0,0,0.45);
    --shadow-purple: 0 0 20px rgba(138,112,255,0.12);
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--text-body);
    font-size: 16px;
}
.stApp { background: var(--bg-page) !important; }
#MainMenu, footer[data-testid="stFooter"] { visibility: hidden; height: 0; }

.block-container {
    padding: 3.5rem 1.5rem 2.5rem !important;
    max-width: 880px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-elevated) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.35) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.35rem 1.1rem 1.75rem !important;
    max-width: 100% !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: var(--text-secondary) !important; font-size: 1rem !important; }
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.15rem 0 !important;
}

.sidebar-brand { font-size: 1.35rem; font-weight: 700; color: var(--text-primary); margin: 0 0 0.25rem 0; letter-spacing: -0.02em; }
.sidebar-brand span { color: var(--accent); }
.sidebar-tagline { font-size: 1rem; color: var(--text-label); margin-bottom: 1.25rem; }
.sidebar-section-title {
    font-size: 0.95rem !important; font-weight: 600 !important;
    color: var(--text-secondary) !important; margin: 0 0 0.65rem 0 !important;
    padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
}

.hero-kicker { font-size: 0.875rem; color: var(--text-label); font-weight: 500; margin-bottom: 0.5rem; }
.hero-title {
    font-family: 'Inter', sans-serif !important; font-size: clamp(1.75rem,3.5vw,2.125rem) !important;
    font-weight: 700 !important; color: var(--text-primary) !important;
    line-height: 1.2 !important; margin: 0 0 0.65rem 0 !important; letter-spacing: -0.02em;
}
.hero-title .accent-title { color: var(--accent); }
.hero-subtitle { font-size: 1rem; color: var(--text-secondary); line-height: 1.65; max-width: 40rem; margin: 0 0 1.5rem 0; }
.section-label { font-size: 0.875rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.65rem; }

.stTextArea textarea {
    background: var(--bg-input) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; color: var(--text-body) !important;
    font-family: 'Inter', sans-serif !important; font-size: 1rem !important; line-height: 1.65 !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-mid) !important;
    box-shadow: 0 0 0 2px rgba(138,112,255,0.25) !important;
}
.stTextArea label { display: none !important; }

.stButton > button[kind="primary"] {
    background: var(--gradient-primary) !important;
    border: 1px solid rgba(138,112,255,0.45) !important;
    color: #fff !important; font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 0.9375rem !important;
    border-radius: var(--radius) !important; padding: 0.65rem 1.35rem !important;
    transition: filter 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: var(--shadow-purple);
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.06);
    box-shadow: 0 0 24px rgba(138,112,255,0.25) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important; font-weight: 500 !important;
    border-radius: var(--radius) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--bg-input) !important;
    border-color: rgba(138,112,255,0.35) !important;
    color: var(--text-primary) !important;
}

.stSlider div[data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important; border: 2px solid var(--bg-card) !important;
}
[data-testid="stFileUploader"] section {
    background: var(--bg-input) !important;
    border: 1px dashed rgba(138,112,255,0.28) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzone"] p { color: var(--text-label) !important; font-size: 0.875rem !important; }

.stProgress > div > div { background: var(--border) !important; border-radius: 999px !important; }
.stProgress > div > div > div { background: var(--accent) !important; border-radius: 999px !important; }
.streamlit-expanderHeader {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--text-secondary) !important; font-size: 0.875rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    border-top: none !important; border-radius: 0 0 var(--radius) var(--radius) !important;
}
.stDownloadButton > button {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important; font-size: 0.875rem !important;
    font-weight: 500 !important; border-radius: var(--radius) !important;
}
.stDownloadButton > button:hover { border-color: var(--accent-mid) !important; color: var(--text-primary) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }

.stats-row { display:flex; gap:0.85rem; flex-wrap:wrap; margin:0.35rem 0 0.15rem; }
.stat-card {
    flex:1; min-width:148px; background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius); padding:1.25rem 1.1rem; text-align:center; box-shadow:var(--shadow-card);
}
.stat-card .val { font-size:1.5rem; font-weight:700; color:var(--text-primary); line-height:1.15; }
.stat-card .lbl { font-size:0.8125rem; font-weight:500; color:var(--text-label); margin-top:0.4rem; }

.summary-box {
    background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius);
    padding:1.35rem 1.5rem; font-size:1rem; line-height:1.7; color:var(--text-body);
    margin-top:0.35rem; box-shadow:var(--shadow-card), inset 4px 0 0 #8a70ff;
}
.ref-box {
    background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius);
    padding:1.2rem 1.35rem; font-size:0.9375rem; line-height:1.7; color:var(--text-secondary);
}
.rouge-wrap { display:flex; gap:0.75rem; flex-wrap:wrap; margin:0.25rem 0 0.35rem; }
.rouge-chip {
    background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius);
    padding:0.85rem 1.1rem; text-align:center; min-width:104px; flex:1; box-shadow:var(--shadow-card);
}
.rouge-val { font-size:1.125rem; font-weight:700; color:var(--accent); }
.rouge-lbl { font-size:0.8125rem; font-weight:500; color:var(--text-label); margin-top:0.35rem; }

.read-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius); padding:1.15rem 1.25rem; box-shadow:var(--shadow-card);
}
.read-bar-bg { background:var(--border); border-radius:999px; height:6px; margin-top:10px; overflow:hidden; }
.read-bar-fill { height:100%; border-radius:999px; }
.read-score { font-size:1.375rem; font-weight:700; line-height:1.1; }
.read-sub { font-size:0.8125rem; font-weight:600; color:var(--text-label); margin-bottom:0.4rem; }
.read-label { font-size:0.875rem; margin-top:0.4rem; font-weight:500; }

.kw-wrap { display:flex; flex-wrap:wrap; gap:0.5rem; margin:0.45rem 0 1rem; }
.kw-pill {
    background:var(--bg-input); border:1px solid var(--border); color:var(--accent);
    border-radius:8px; padding:5px 12px; font-size:0.8125rem; font-weight:500;
}
.kw-highlight { background:rgba(138,112,255,0.22); color:#c4b5fd; border-radius:4px; padding:1px 4px; font-weight:600; }
.article-view {
    font-size:0.9375rem; line-height:1.65; color:var(--text-body);
    background:var(--bg-input); border:1px solid var(--border);
    border-radius:var(--radius); padding:1.15rem 1.25rem; max-height:300px; overflow-y:auto;
}
.article-view::-webkit-scrollbar { width:6px; }
.article-view::-webkit-scrollbar-thumb { background:rgba(138,112,255,0.35); border-radius:999px; }

.cat-pill { display:inline-block; padding:5px 12px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.cat-business     { background:#422006; color:#fcd34d; border:1px solid #713f12; }
.cat-tech         { background:#2a1f4a; color:#c4b5fd; border:1px solid #5b47a8; }
.cat-sport        { background:#14532d; color:#86efac; border:1px solid #166534; }
.cat-politics     { background:#450a0a; color:#fca5a5; border:1px solid #7f1d1d; }
.cat-entertainment{ background:#351d5c; color:#ddd6fe; border:1px solid #6d52a8; }

.thin-div { height:1px; background:var(--border); margin:1.5rem 0; }
.stCaption { color:var(--text-label) !important; font-size:0.875rem !important; }

/* Chart containers */
.chart-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius); padding:1.2rem 1.4rem;
    box-shadow:var(--shadow-card); margin-bottom:0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#c8c8d4"),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="#2a2a36", zerolinecolor="#2a2a36", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#2a2a36", zerolinecolor="#2a2a36", tickfont=dict(size=11)),
)


def plot_sentence_scores(article, method_key, n):
    """Bar chart: TF-IDF score per sentence, selected ones highlighted."""
    sentences = sent_tokenize(article)
    if len(sentences) < 2:
        return None

    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        mat = vectorizer.fit_transform(sentences)
    except Exception:
        return None

    scores = np.array(mat.sum(axis=1)).flatten()
    top_idx = set(np.argsort(scores)[-n:])

    colors = ["#8a70ff" if i in top_idx else "#2a2a36" for i in range(len(sentences))]
    borders = ["#8a70ff" if i in top_idx else "#3a3a48" for i in range(len(sentences))]
    labels  = [f"S{i+1}" for i in range(len(sentences))]
    hover   = [f"<b>Sentence {i+1}</b><br>Score: {scores[i]:.4f}<br>{'✓ Selected' if i in top_idx else 'Not selected'}<br><br>{sentences[i][:120]}..." for i in range(len(sentences))]

    fig = go.Figure(go.Bar(
        x=labels, y=scores,
        marker=dict(color=colors, line=dict(color=borders, width=1.5)),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Sentence TF‑IDF Scores  <span style='font-size:12px;color:#9494a3'>· purple = selected</span>", font=dict(size=14)),
        height=300,
        showlegend=False,
    )
    return fig


def plot_word_frequency(article, top_n=15):
    """Horizontal bar chart of top N word frequencies."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(article.lower())
    freq = {}
    for w in words:
        w = re.sub(r'[^a-zA-Z]', '', w)
        if w and w not in stop_words and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    if not freq:
        return None

    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words_list  = [w for w, _ in reversed(sorted_freq)]
    counts      = [c for _, c in reversed(sorted_freq)]

    max_c = max(counts)
    bar_colors = [
        f"rgba(138,112,255,{0.4 + 0.6 * (c / max_c):.2f})"
        for c in counts
    ]

    fig = go.Figure(go.Bar(
        x=counts, y=words_list,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color="rgba(138,112,255,0.6)", width=1)),
        hovertemplate="<b>%{y}</b>: %{x} times<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"Top {top_n} Word Frequencies  <span style='font-size:12px;color:#9494a3'>· after stopword removal</span>", font=dict(size=14)),
        height=420,
        showlegend=False,
        xaxis_title="Frequency",
    )
    return fig


def plot_rouge_comparison(article, reference, n):
    """Grouped bar chart comparing ROUGE scores across all 3 methods."""
    methods    = ['TF-IDF', 'TextRank', 'Frequency']
    method_keys = ['tfidf', 'textrank', 'frequency']
    metrics    = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    colors     = ['#8a70ff', '#5eead4', '#fbbf24']

    all_scores = {}
    for mk in method_keys:
        gen = summarize(article, method=mk, n=n)
        sc  = rouge_module.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True).score(reference, gen)
        all_scores[mk] = [
            round(sc['rouge1'].fmeasure, 4),
            round(sc['rouge2'].fmeasure, 4),
            round(sc['rougeL'].fmeasure, 4),
        ]

    fig = go.Figure()
    for i, (mk, label) in enumerate(zip(method_keys, methods)):
        fig.add_trace(go.Bar(
            name=label,
            x=metrics,
            y=all_scores[mk],
            marker=dict(color=colors[i], line=dict(color=colors[i], width=1)),
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROUGE Score Comparison  <span style='font-size:12px;color:#9494a3'>· all 3 methods</span>", font=dict(size=14)),
        barmode='group',
        height=340,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8c8d4", size=12),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# OTHER HELPERS
# ══════════════════════════════════════════════════════════════

def extract_keywords(text, top_n=12):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
    try:
        tfidf = vectorizer.fit_transform(sentences)
        scores = np.array(tfidf.sum(axis=0)).flatten()
        vocab  = vectorizer.get_feature_names_out()
        top_idx = np.argsort(scores)[-top_n:][::-1]
        return [vocab[i] for i in top_idx]
    except Exception:
        return []

def highlight_keywords(text, keywords):
    if not keywords:
        return text
    pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
    return re.sub(pattern, r'<span class="kw-highlight">\1</span>', text, flags=re.IGNORECASE)

def get_readability(text):
    try:
        return round(textstat.flesch_reading_ease(text), 1)
    except Exception:
        return 0.0

def readability_info(score):
    if score >= 70: return ("Easy to read", "#4ade80", "#4ade80")
    if score >= 50: return ("Moderate",     "#fbbf24", "#fbbf24")
    return ("Complex", "#f87171", "#f87171")

def compute_rouge(reference, generated):
    scorer = rouge_module.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    s = scorer.score(reference, generated)
    return {
        'ROUGE-1': round(s['rouge1'].fmeasure, 4),
        'ROUGE-2': round(s['rouge2'].fmeasure, 4),
        'ROUGE-L': round(s['rougeL'].fmeasure, 4),
    }

def read_pdf(f):
    if PyPDF2 is None: return ""
    r = PyPDF2.PdfReader(io.BytesIO(f.read()))
    return "".join([p.extract_text() or "" for p in r.pages]).strip()

def read_docx(f):
    if Document is None: return ""
    doc = Document(io.BytesIO(f.read()))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<p class="sidebar-brand">📰 NEWS<span>SNAP</span></p>'
        '<p class="sidebar-tagline">NLP Text Summarizer · 21CSE356T</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sidebar-section-title">⚙️ Settings</p>', unsafe_allow_html=True)
    method = st.radio(
        "Method", ["TF-IDF", "TextRank", "Frequency"],
        captions=["Sentence scoring", "Graph · PageRank", "Frequency baseline"],
        label_visibility="collapsed",
    )
    n_sentences = st.slider("📏 Summary length (sentences)", 2, 7, 3)

    st.divider()
    st.markdown('<p class="sidebar-section-title">🗂️ Dataset</p>', unsafe_allow_html=True)
    load_btn = st.button("🎲 Load Random Article", use_container_width=True, type="secondary")
    if 'category' in st.session_state:
        cat   = st.session_state['category']
        fname = st.session_state['filename']
        st.markdown(
            f'<span class="cat-pill cat-{cat}">{cat}</span> '
            f'<span style="color:#9494a3;font-size:0.8125rem;margin-left:8px">{fname}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<p class="sidebar-section-title">📎 Upload</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF or DOCX", type=["pdf","docx"], label_visibility="collapsed")
    if uploaded_file:
        with st.spinner("Reading file…"):
            extracted = read_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else read_docx(uploaded_file)
        if extracted:
            st.session_state['article'] = extracted
            for k in ['reference','category','filename']:
                st.session_state.pop(k, None)
            st.success(f"Loaded `{uploaded_file.name}`")
        else:
            st.error("Could not extract text from this file.")


# ══════════════════════════════════════════════════════════════
# LOAD RANDOM
# ══════════════════════════════════════════════════════════════

if load_btn:
    art, ref, cat, fname = get_random_article()
    st.session_state.update({'article':art,'reference':ref,'category':cat,'filename':fname})


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

st.markdown(
    '<p class="hero-kicker">SRM Institute of Science and Technology · 21CSE356T</p>'
    '<h1 class="hero-title">NewsSnap <span class="accent-title">– NLP Text Summarizer</span></h1>'
    '<p class="hero-subtitle">Extractive summarization on news text. Choose TF‑IDF, TextRank, or Frequency '
    'and generate a concise summary with readability insight, keyword analysis, and ROUGE metrics.</p>',
    unsafe_allow_html=True,
)

st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)

st.markdown('<p class="section-label">Article input</p>', unsafe_allow_html=True)
article_input = st.text_area(
    "Article",
    value=st.session_state.get('article',''),
    height=280,
    placeholder="Paste an article, use Load Random Article in the sidebar, or upload a PDF / DOCX…",
    label_visibility="collapsed",
)

summarize_btn = st.button("Summarize", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════

if summarize_btn:
    if not article_input.strip():
        st.warning("Add or load article text before summarizing.")
    else:
        with st.spinner("Analyzing…"):
            method_key  = method.lower().replace('-','')
            article_clean = "\n".join(article_input.strip().split("\n")[1:])
            generated = summarize(article_clean, method=method_key, n=n_sentences)
            keywords    = extract_keywords(article_input)
            read_orig   = get_readability(article_input)
            read_summ   = get_readability(generated)
            orig_words  = len(article_input.split())
            summ_words  = len(generated.split())
            compression = round(summ_words / orig_words * 100, 1) if orig_words else 0.0

        st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)

        # ── Stats ────────────────────────────────────────────────
        st.markdown('<p class="section-label">Statistics</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card"><div class="val">{orig_words:,}</div><div class="lbl">Original words</div></div>
            <div class="stat-card"><div class="val">{summ_words:,}</div><div class="lbl">Summary words</div></div>
            <div class="stat-card"><div class="val">{compression}%</div><div class="lbl">Compression</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)

        # ── Summary ──────────────────────────────────────────────
        sum_head, sum_dl = st.columns([4,1])
        with sum_head:
            st.markdown(f'<p class="section-label">Generated summary · {method}</p>', unsafe_allow_html=True)
        with sum_dl:
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("Download", data=generated, file_name="summary.txt",
                               mime="text/plain", use_container_width=True)
        st.markdown(f'<div class="summary-box">{generated}</div>', unsafe_allow_html=True)

        st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # GRAPH 1 — Sentence Score Bar Chart
        # ══════════════════════════════════════════════════════
        st.markdown('<p class="section-label">📊 Graph 1 · Sentence TF‑IDF Scores</p>', unsafe_allow_html=True)
        st.caption("Each bar is one sentence. Purple bars are the sentences selected for your summary.")
        fig1 = plot_sentence_scores(article_input, method_key, n_sentences)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Not enough sentences to plot.")

        st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # GRAPH 2 — Word Frequency Chart
        # ══════════════════════════════════════════════════════
        st.markdown('<p class="section-label">📊 Graph 2 · Word Frequency Distribution</p>', unsafe_allow_html=True)
        st.caption("Top 15 most frequent words after stopword removal. Darker bars = higher frequency.")
        fig2 = plot_word_frequency(article_input)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Could not compute word frequencies.")

        st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)

        # ── Readability ──────────────────────────────────────
        st.markdown('<p class="section-label">Readability · Flesch Reading Ease</p>', unsafe_allow_html=True)
        lbl_o, col_o, _ = readability_info(read_orig)
        lbl_s, col_s, _ = readability_info(read_summ)
        bar_o = min(max(int(read_orig), 0), 100)
        bar_s = min(max(int(read_summ), 0), 100)
        diff  = round(read_summ - read_orig, 1)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="read-card">
                <div class="read-sub">Original article</div>
                <div class="read-score" style="color:{col_o}">{read_orig}</div>
                <div class="read-label" style="color:{col_o}">{lbl_o}</div>
                <div class="read-bar-bg"><div class="read-bar-fill" style="width:{bar_o}%;background:{col_o}"></div></div>
            </div>""", unsafe_allow_html=True)
        with r2:
            arrow = "↑" if diff > 0 else "↓"
            st.markdown(f"""
            <div class="read-card">
                <div class="read-sub">Generated summary</div>
                <div class="read-score" style="color:{col_s}">{read_summ}</div>
                <div class="read-label" style="color:{col_s}">{lbl_s}
                  <span style="font-size:0.875rem;color:#9494a3"> · {arrow} {abs(diff)} vs original</span>
                </div>
                <div class="read-bar-bg"><div class="read-bar-fill" style="width:{bar_s}%;background:{col_s}"></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)

        # ── Keywords ─────────────────────────────────────────
        st.markdown('<p class="section-label">Top keywords · TF‑IDF</p>', unsafe_allow_html=True)
        pills = "".join([f'<span class="kw-pill">{k}</span>' for k in keywords])
        st.markdown(f'<div class="kw-wrap">{pills}</div>', unsafe_allow_html=True)

        with st.expander("View article with keywords highlighted"):
            highlighted = highlight_keywords(article_input, keywords)
            st.markdown(f'<div class="article-view">{highlighted}</div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # ROUGE + GRAPH 3
        # ══════════════════════════════════════════════════════
        if ('reference' in st.session_state and
                article_input.strip() == st.session_state.get('article','').strip()):

            st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-label">ROUGE evaluation</p>', unsafe_allow_html=True)

            rouge_scores = compute_rouge(st.session_state['reference'], generated)
            chips = "".join([
                f'<div class="rouge-chip"><div class="rouge-val">{v:.3f}</div><div class="rouge-lbl">{k}</div></div>'
                for k, v in rouge_scores.items()
            ])
            st.markdown(f'<div class="rouge-wrap">{chips}</div>', unsafe_allow_html=True)

            # ── Graph 3 ──────────────────────────────────────
            st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-label">📊 Graph 3 · ROUGE Score Comparison · All 3 Methods</p>', unsafe_allow_html=True)
            st.caption("Compares TF-IDF, TextRank, and Frequency on ROUGE-1, ROUGE-2, ROUGE-L against the BBC reference summary.")

            with st.spinner("Running all 3 methods for comparison…"):
                fig3 = plot_rouge_comparison(article_input, st.session_state['reference'], n_sentences)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

            with st.expander("View BBC reference summary"):
                st.markdown(f'<div class="ref-box">{st.session_state["reference"]}</div>', unsafe_allow_html=True)


st.markdown('<div class="thin-div"></div>', unsafe_allow_html=True)
st.caption("SRM Institute of Science and Technology · 21CSE356T Natural Language Processing · BBC News Dataset")