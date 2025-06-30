import streamlit as st
import pickle
import gdown
import zipfile
import io
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 🔽 CONFIG ------------------ #
st.set_page_config(page_title="Medicine Recommender", layout="centered")

# ------------------ 🔽 UTILS ------------------ #

def load_pickle_from_drive(drive_url, filename):
    gdown.download(drive_url, output=filename, quiet=False, use_cookies=False)
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
        "https://drive.google.com/uc?id=1Q1d2ktBMd1FXMbo0McD6zVFhlqs3p4dY", 
        "grouped.pkl"
    )
    similarity_matrix = load_pickle_from_drive(
        "https://drive.google.com/uc?id=1d0RFiRioEy4EWN4M2tofRLyMvWcO9g3D", 
        "similarity_matrix.pkl"
    )

    tfidf = load_pickle_from_zip("tfidf.pkl.zip", "tfidf.pkl")
    combined_features = load_pickle_from_zip("combined.pkl.zip", "combined.pkl")

    return grouped, similarity_matrix, tfidf, combined_features

grouped, similarity_matrix, tfidf, combined_features = load_all()

# ------------------ 🔍 Recommendation Function ------------------ #
def recommend(drug_name, top_n=5):
    selected = grouped[grouped['drugName'].str.lower() == drug_name.lower()]
    if selected.empty:
        return "❌ No matching drug found."

    index = selected.index[0]
    sim_scores = similarity_matrix[index]
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]

    results = []
    for i in similar_indices:
        results.append({
            "🧪 Medicine": grouped.iloc[i]['drugName'],
            "📋 Condition": grouped.iloc[i]['condition'],
            "⭐ Rating": round(grouped.iloc[i]['rating'], 2),
            "📊 Similarity": round(sim_scores[i], 3),
            "🗣 Review": grouped.iloc[i]['review'][:300] + "..."
        })
    return results

# ------------------ 🎯 Streamlit UI ------------------ #
st.title("💊 Personalised Medicine Recommender")

user_input = st.text_input("Enter a medicine name (e.g., Afatinib):")

if user_input:
    with st.spinner("🔍 Finding similar medicines..."):
        results = recommend(user_input)
    
    if isinstance(results, str):
        st.error(results)
    else:
        for rec in results:
            st.markdown("### 🧪 " + rec["🧪 Medicine"])
            st.markdown("**📋 Condition:** " + rec["📋 Condition"])
            st.markdown(f"**⭐ Rating:** {rec['⭐ Rating']}")
            st.markdown(f"**📊 Similarity:** {rec['📊 Similarity']}")
            st.markdown("🗣 _" + rec["🗣 Review"] + "_")
            st.markdown("---")
