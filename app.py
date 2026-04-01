import streamlit as st
from summarizer import summarize
from evaluate import get_random_article

st.set_page_config(page_title="NewsSnap AI", layout="wide", page_icon="⚡")

st.markdown("""
<style>
.stButton > button {
    width: 100%;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem !important;
    transition: all 0.3s ease !important;
}
.stTextArea textarea {
    border-radius: 12px !important;
    font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;font-size:3rem'>⚡ NewsSnap AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;opacity:0.5;letter-spacing:3px;font-size:0.85rem'>BBC NEWS SUMMARIZER · 21CSE356T NLP</p>", unsafe_allow_html=True)
st.divider()

# ── CONTROLS ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### ⚙️ Settings")

    method = st.radio(
        "Summarization Method",
        options=["TF-IDF", "TextRank", "Frequency"],
        captions=["Best accuracy", "Graph-based", "Baseline"],
        index=0
    )

    n_sentences = st.slider("Summary Length (sentences)", 2, 7, 3)

    st.divider()

    st.markdown("### 📁 Dataset")
    load_btn = st.button("🎲 Load Random Article", use_container_width=True)

    if 'category' in st.session_state:
        category_colors = {
            "business": "🟡",
            "tech": "🔵",
            "sport": "🟢",
            "politics": "🩷",
            "entertainment": "🟣"
        }
        cat = st.session_state['category']
        icon = category_colors.get(cat, "⚪")
        st.success(f"{icon} **{cat.upper()}** · `{st.session_state['filename']}`")

with col_right:
    st.markdown("### 📰 Article Input")

    if load_btn:
        article, reference, category, filename = get_random_article()
        st.session_state['article'] = article
        st.session_state['reference'] = reference
        st.session_state['category'] = category
        st.session_state['filename'] = filename

    article_input = st.text_area(
        "",
        value=st.session_state.get('article', ''),
        height=320,
        placeholder="Paste any news article here, or load one from the dataset →",
        label_visibility="collapsed"
    )

    summarize_btn = st.button("⚡ Summarize Now", use_container_width=True, type="primary")

    if summarize_btn:
        if article_input.strip():
            with st.spinner("Summarizing..."):
                method_key = method.lower().replace('-', '')
                generated = summarize(article_input, method=method_key, n=n_sentences)

            st.divider()

            # ── METRICS ─────────────────────────────────────────────────
            orig_words = len(article_input.split())
            summ_words = len(generated.split())
            ratio = f"{summ_words / orig_words:.1%}"

            m1, m2, m3 = st.columns(3)
            m1.metric("📄 Original Words", orig_words)
            m2.metric("✂️ Summary Words", summ_words)
            m3.metric("📉 Compression", ratio)

            st.divider()

            # ── SUMMARY OUTPUT ──────────────────────────────────────────
            st.markdown(f"### ✦ Generated Summary · `{method}`")
            st.info(generated)

            # ── REFERENCE SUMMARY ───────────────────────────────────────
            if 'reference' in st.session_state and article_input.strip() == st.session_state.get('article', '').strip():
                st.markdown("### 📄 Reference Summary · BBC Dataset")
                st.success(st.session_state['reference'])

        else:
            st.warning("⚠️ Please paste an article or load one from the dataset.")

# ── FOOTER ──────────────────────────────────────────────────────────────
st.divider()
st.markdown("<p style='text-align:center;opacity:0.3;font-size:0.75rem;letter-spacing:2px'>SRM INSTITUTE OF SCIENCE AND TECHNOLOGY · NLP PROJECT</p>", unsafe_allow_html=True)