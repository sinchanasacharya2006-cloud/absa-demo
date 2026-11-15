import streamlit as st
import torch, spacy, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "sinchana-ai/distilbert-absa"
COLORS = {"positive": "#2e7d32", "negative": "#c62828", "neutral": "#757575"}
STOP_ASPECTS = {"thing", "time", "way", "lot", "bit", "day", "year", "month",
                "week", "one", "something", "anything", "everything", "people",
                "monday", "tuesday", "wednesday", "thursday", "friday",
                "saturday", "sunday", "laptop", "computer"}

@st.cache_resource
def load():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    return nlp, tok, model

nlp, tok, model = load()

def extract_aspects(sentence):
    aspects = []
    for chunk in nlp(sentence).noun_chunks:
        if chunk.root.pos_ not in ("NOUN", "PROPN"):
            continue
        if chunk.root.lemma_.lower() in STOP_ASPECTS:
            continue
        words = [t.text.lower() for t in chunk
                 if t.dep_ == "compound" or t == chunk.root]
        aspects.append(" ".join(words))
    return list(dict.fromkeys(aspects))

def analyze(review):
    results = []
    for sent in nlp(review).sents:
        for asp in extract_aspects(sent.text):
            enc = tok(asp, sent.text, return_tensors="pt",
                      truncation=True, max_length=128)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)[0]
            label = model.config.id2label[int(probs.argmax())]
            results.append({"aspect": asp, "sentiment": label,
                            "confidence": round(float(probs.max()), 2)})
    return results

st.title("Product Review Aspect & Sentiment Analyzer")
st.write("Paste one or more reviews (one per line).")

text = st.text_area("Reviews", height=180,
    value="The battery life is great but the screen is dull.\nThe keyboard feels cheap but the price is reasonable.")

if st.button("Analyze"):
    reviews = [r.strip() for r in text.split("\n") if r.strip()]
    rows = []
    for rev in reviews:
        res = analyze(rev)
        rows.extend(res)
        tags = " ".join(
            f'<span style="background:{COLORS[r["sentiment"]]};color:white;'
            f'padding:2px 8px;border-radius:10px;margin-right:4px;'
            f'opacity:{1.0 if r["confidence"] >= 0.6 else 0.5}">'
            f'{r["aspect"]}: {r["sentiment"]}</span>' for r in res)
        st.markdown(f"**{rev}**")
        st.markdown(tags or "_no aspects found_", unsafe_allow_html=True)
    if rows:
        st.caption("Faded tags = low confidence (below 0.6)")
        df_res = pd.DataFrame(rows)
        counts = (df_res.groupby(["aspect", "sentiment"]).size()
                  .unstack(fill_value=0))
        st.subheader("Aspect summary")
        st.bar_chart(counts)
        st.dataframe(df_res)
