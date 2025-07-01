import streamlit as st
import pickle
import zipfile
import gdown
import os
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 🔽 UTILITY FUNCTIONS ------------------ #

def load_pickle_from_drive(file_id, filename):
    if not os.path.exists(filename):
        gdown.download(id=file_id, output=filename, quiet=False, fuzzy=True)
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_pickle_from_zip(zip_filename, pkl_filename):
    with zipfile.ZipFile(zip_filename, 'r') as z:
        with z.open(pkl_filename) as f:
            return pickle.load(f)

# ------------------ 🔽 LOAD PICKLES ------------------ #

@st.cache_data
def load_all():
    grouped = load_pickle_from_drive(
        "1mLRUtjl2PubY3Ago0iAlrRtaUcROVFE5",  # grouped.pkl
        "grouped.pkl"
    )
    similarity_matrix = load_pickle_from_drive(
        "1D8R85eUVDvNwHpS_M_8_gc-nlhunpZx0",  # similarity_matrix.pkl
        "similarity_matrix.pkl"
    )
    tfidf = load_pickle_from_zip("tfidf.pkl.zip", "tfidf.pkl")
    combined_features = load_pickle_from_zip("combined.pkl.zip", "combined.pkl")
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
