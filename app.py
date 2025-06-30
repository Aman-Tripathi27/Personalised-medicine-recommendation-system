import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import zipfile
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 🔽 Download from Google Drive using full link ------------------ #
def load_pickle_from_drive(drive_url, filename):
    gdown.download(drive_url, output=filename, quiet=False, fuzzy=True)
    with open(filename, "rb") as f:
        return pickle.load(f)

# ------------------ 🔽 Unzip and load local .pkl from zip ------------------ #
def load_pickle_from_zip(zip_filename, pkl_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extract(pkl_filename)
    with open(pkl_filename, "rb") as f:
        return pickle.load(f)

# ------------------ 🔽 Load All Pickles ------------------ #
@st.cache_data
def load_all():
    grouped = load_pickle_from_drive(
        "https://drive.google.com/uc?id=1Q1d2ktBMd1FXMbo0McD6zVFhlqs3p4dY", "grouped.pkl"
    )
    similarity_matrix = load_pickle_from_drive(
        "https://drive.google.com/uc?id=1d0RFiRioEy4EWN4M2tofRLyMvWcO9g3D", "similarity_matrix.pkl"
    )

    tfidf = load_pickle_from_zip("tfidf.pkl.zip", "tfidf.pkl")
    combined_features = load_pickle_from_zip("combined_features.pkl.zip", "combined_features.pkl")

    return grouped, similarity_matrix, tfidf, combined_features

grouped, similarity_matrix, tfidf, combined_features = load_all()

# ------------------ 🔍 Recommendation Function ------------------ #
def recommend(drug_name, top_n=5):
    selected = grouped[grouped['drugName'].str.lower() == drug_name.lower()]
    if selected.empty:
        return "❌ No matching drug found."

    index = selected.index[0]
    sim_scores = cosine_similarity(combined_features[index], combined_features).flatten()
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

# ------------------ 🧠 Streamlit UI ------------------ #
st.title("💊 Personalized Medicine Recommendation")

drug_input = st.text_input("Enter a medicine name:")

if st.button("Recommend") and drug_input:
    results = recommend(drug_input)
    if isinstance(results, str):
        st.error(results)
    else:
        for rec in results:
            st.markdown(f"""---
**🧪 Medicine:** {rec['🧪 Medicine']}  
**📋 Condition:** {rec['📋 Condition']}  
**⭐ Rating:** {rec['⭐ Rating']}  
**📊 Similarity:** {rec['📊 Similarity']}  
**🗣 Review:** {rec['🗣 Review']}  
""")
