import streamlit as st
import pickle
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 🔽 UTILITY FUNCTION TO LOAD PICKLE FROM HUGGING FACE ------------------ #

def load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.loads(response.content)

# ------------------ 🔽 LOAD ALL FILES ------------------ #

@st.cache_data
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
