import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 🔽 HELPER TO LOAD FROM DRIVE ------------------ #
def load_pickle_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    return pickle.load(io.BytesIO(response.content))

# ------------------ 🔽 LOAD PICKLES ------------------ #
@st.cache_data
def load_all():
    grouped = load_pickle_from_drive("1Q1d2ktBMd1FXMbo0McD6zVFhlqs3p4dY")  # grouped.pkl
    similarity_matrix = load_pickle_from_drive("1d0RFiRioEy4EWN4M2tofRLyMvWcO9g3D")  # similarity_matrix.pkl

    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("combined_features.pkl", "rb") as f:
        combined_features = pickle.load(f)

    return grouped, similarity_matrix, tfidf, combined_features

grouped, similarity_matrix, tfidf, combined_features = load_all()

# ------------------ 🔽 RECOMMENDATION FUNCTION ------------------ #
def recommend(drug_name, top_n=5):
    selected = grouped[grouped['drugName'].str.lower() == drug_name.lower()]
    if selected.empty:
        return []

    index = selected.index[0]
    sim_scores = similarity_matrix[index]
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]

    recommendations = []
    for i in similar_indices:
        rec = {
            "🧪 Medicine": grouped.iloc[i]['drugName'],
            "📋 Condition": grouped.iloc[i]['condition'],
            "⭐ Rating": round(grouped.iloc[i]['rating'], 2),
            "📊 Similarity": round(sim_scores[i], 3),
            "🗣 Review": grouped.iloc[i]['review'][:300] + "..."
        }
        recommendations.append(rec)

    return recommendations

# ------------------ 🔽 STREAMLIT UI ------------------ #
st.title("💊 Personalized Medicine Recommender")

drug_input = st.text_input("Enter a medicine name (e.g., 'Afatinib')")

if st.button("Recommend"):
    if not drug_input.strip():
        st.warning("⚠️ Please enter a medicine name.")
    else:
        results = recommend(drug_input.strip())
        if not results:
            st.error("❌ No similar medicines found.")
        else:
            for rec in results:
                st.markdown(f"### 🧪 Medicine: {rec['🧪 Medicine']}")
                st.markdown(f"📋 **Condition**: {rec['📋 Condition']}")
                st.markdown(f"⭐ **Rating**: {rec['⭐ Rating']}")
                st.markdown(f"📊 **Similarity**: {rec['📊 Similarity']}")
                st.markdown(f"🗣 **Review**: _{rec['🗣 Review']}_")
                st.markdown("---")
