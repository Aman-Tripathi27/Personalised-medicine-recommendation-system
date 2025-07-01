import streamlit as st
import pickle
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 🔽 UTILITY FUNCTIONS ------------------ #

def download_pickle_from_gdrive(url, filename):
    """Download a pickle file from Google Drive using requests."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {filename} from Google Drive.")
    with open(filename, "wb") as f:
        f.write(response.content)
    with open(filename, "rb") as f:
        return pickle.load(f)

# ------------------ 🔽 LOAD PICKLES ------------------ #

@st.cache_data
def load_all():
    grouped = download_pickle_from_gdrive(
        "https://drive.google.com/uc?export=download&id=1D8R85eUVDvNwHpS_M_8_gc-nlhunpZx0",
        "grouped.pkl"
    )
    similarity_matrix = download_pickle_from_gdrive(
        "https://drive.google.com/uc?export=download&id=1d0RFiRioEy4EWN4M2tofRLyMvWcO9g3D",
        "similarity_matrix.pkl"
    )

    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("combined_features.pkl", "rb") as f:
        combined_features = pickle.load(f)

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

# ------------------ 🔽 MAIN UI ------------------ #

st.set_page_config(page_title="Medicine Recommendation", layout="wide")
st.title("🧠 Personalized Medicine Recommendation System")

# Load data
grouped, similarity_matrix, tfidf, combined_features = load_all()

drug_name = st.text_input("🔍 Enter a medicine name:", "")
if drug_name:
    with st.spinner("🔎 Finding similar medicines..."):
        output = recommend(drug_name)
        if isinstance(output, str):
            st.warning(output)
        else:
            for rec in output:
                st.markdown(f"""
                **🧪 Medicine:** {rec['🧪 Medicine']}  
                **📋 Condition:** {rec['📋 Condition']}  
                **⭐ Rating:** {rec['⭐ Rating']}  
                **📊 Similarity:** {rec['📊 Similarity']}  
                **🗣 Review:** {rec['🗣 Review']}  
                ---
                """)
