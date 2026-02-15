import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Narrative Detect", layout="wide")

st.title("Narrative Detect (prototype)")

st.write("Offline-first narrative detection + idea generation.")

sample = Path(__file__).resolve().parents[1] / "data" / "sample_posts.jsonl"

input_path = st.text_input("JSONL path", value=str(sample))
method = st.selectbox("Method", ["tfidf", "semantic"], index=0)
k = st.slider("k (tfidf mode)", 2, 20, 6)

if st.button("Run detection"):
    from narrative_detect.detect import detect_narratives

    report = detect_narratives(Path(input_path), method=method, k=k)
    st.subheader("Narratives")
    for n in report["narratives"]:
        st.markdown(f"### {n['label']}")
        st.write({k: n[k] for k in ["size", "trend_score", "keywords"]})
        st.write("Representative posts")
        for p in n["representative_posts"]:
            st.write("- ", p)

    st.download_button(
        "Download report.json",
        data=json.dumps(report, indent=2),
        file_name="report.json",
        mime="application/json",
    )
