import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import pickle
import zipfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# 📥 Step 1: Download zipped pickle from Google Drive
@st.cache_data
def download_and_extract_pickle():
    zip_file = "med.pkl.zip"
    pkl_file = "med.pkl"
    # Google Drive shareable ID (replace if needed)
    gdrive_url = "https://drive.google.com/uc?id=1shXUL3RkrSz_NDYsqFd2XU_Y3DNvMkOF"

    if not os.path.exists(zip_file):
        gdown.download(gdrive_url, zip_file, quiet=False)

    if not os.path.exists(pkl_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extract(pkl_file)

    with open(pkl_file, 'rb') as f:
        med = pickle.load(f)
    return med

# ⚙️ Step 2: Feature engineering
@st.cache_data
def compute_features(med):
    med['text'] = med['drugName'].astype(str) + " " + med['condition'].astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(med['text'])
    scaler = MinMaxScaler()
    rating_vector = scaler.fit_transform(med[['rating']])
    combined_features = hstack([tfidf_matrix, rating_vector])
    return csr_matrix(combined_features)

# 🔍 Step 3: Recommendation logic
def recommend(drug_name, med, combined_features, top_n=5):
    indices = med[med['drugName'].str.lower() == drug_name.lower()].index.tolist()
    if not indices:
        return []
    index = indices[0]
    sim_scores = cosine_similarity(combined_features[index], combined_features).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:top_n + 1]

    results = []
    for i in similar_indices:
        row = med.iloc[i]
        results.append({
            '🧪 Medicine': row['drugName'],
            '📋 Condition': row['condition'],
            '⭐ Rating': row['rating'],
            '🗣 Review': row['review'][:300] + "..." if len(row['review']) > 300 else row['review'],
            '📊 Similarity Score': round(sim_scores[i], 3)
        })
    return results

# 🌐 Streamlit UI
st.title("💊 Personalized Medicine Recommender")

med = download_and_extract_pickle()
combined_features = compute_features(med)

drug_list = sorted(med['drugName'].dropna().unique())
selected_drug = st.selectbox("Select a medicine to find similar ones:", drug_list)

if st.button("Recommend"):
    results = recommend(selected_drug, med, combined_features)
    if not results:
        st.error("No similar medicines found.")
    else:
        for r in results:
            st.markdown(f"""
            ---
            🧪 **Medicine**: `{r['🧪 Medicine']}`  
            📋 **Condition**: _{r['📋 Condition']}_  
            ⭐ **Rating**: {r['⭐ Rating']}  
            📊 **Similarity**: {r['📊 Similarity Score']}  
            🗣 **Review**: {r['🗣 Review']}
            """)
