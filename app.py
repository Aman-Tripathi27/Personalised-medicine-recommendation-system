import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ 🔽 DOWNLOAD FROM HUGGINGFACE ------------------ #
@st.cache_resource
def load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.loads(response.content)

# ------------------ 🔽 LOAD ALL FILES ------------------ #
@st.cache_resource
def load_all():
    grouped = load_pickle_from_url("https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/grouped.pkl")
    similarity_matrix = load_pickle_from_url("https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/similarity_matrix.pkl")
    tfidf = load_pickle_from_url("https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/tfidf.pkl")
    combined_features = load_pickle_from_url("https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/combined_features.pkl")
    return grouped, similarity_matrix, tfidf, combined_features

# ------------------ 🔍 RECOMMENDATION FUNCTION ------------------ #
def recommend(drug_name, top_n=5):
    selected = grouped[grouped['drugName'].str.lower() == drug_name.lower()]
    if selected.empty:
        return f"❌ No match found for: {drug_name}"
    index = selected.index[0]
    sim_scores = cosine_similarity(combined_features[index], combined_features).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]

    results = []
    for i in similar_indices:
        rec = {
            "🧪 Medicine": grouped.iloc[i]['drugName'],
            "📋 Condition": grouped.iloc[i]['condition'],
            "⭐ Rating": round(grouped.iloc[i]['rating'], 2),
            "📊 Similarity": round(sim_scores[i], 3),
            "🗣 Review": grouped.iloc[i]['review'][:300] + "..."
        }
        results.append(rec)
    return results

# ------------------ 🎛️ STREAMLIT UI ------------------ #
st.set_page_config(page_title="Medicine Recommendation", layout="wide")
st.title("🧠 Personalized Medicine Recommendation System")

# Load data
with st.spinner("📦 Loading model and data..."):
    grouped, similarity_matrix, tfidf, combined_features = load_all()

# Dropdown for drug selection
drug_list = sorted(grouped['drugName'].dropna().unique())
drug_name = st.selectbox("🔍 Select a medicine:", drug_list)

# Show recommendations
if drug_name:
    with st.spinner("🔎 Finding similar medicines..."):
        output = recommend(drug_name)
        if isinstance(output, str):
            st.warning(output)
        else:
            for rec in output:
                st.markdown(
                    f"""
                    <div style='font-size: 14px; line-height: 1.6; border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 10px;'>
                        <b>🧪 Medicine:</b> {rec['🧪 Medicine']}<br>
                        <b>📋 Condition:</b> {rec['📋 Condition']}<br>
                        <b>⭐ Rating:</b> {rec['⭐ Rating']}<br>
                        <b>📊 Similarity:</b> {rec['📊 Similarity']}<br>
                        <b>🗣 Review:</b> {rec['🗣 Review']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
