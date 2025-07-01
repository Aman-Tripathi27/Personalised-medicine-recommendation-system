import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 🔽 Download Helper ------------------ #
def download_file(url, filename, is_pickle=True):
    response = requests.get(url)
    response.raise_for_status()
    if is_pickle:
        return pickle.loads(response.content)
    else:
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename

# ------------------ 🔽 Load Data ------------------ #
@st.cache_data
def load_all():
    grouped = download_file(
        "https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/grouped.pkl",
        "grouped.pkl"
    )
    similarity_matrix = download_file(
        "https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/similarity_matrix.pkl",
        "similarity_matrix.pkl"
    )
    tfidf = download_file(
        "https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/tfidf.pkl",
        "tfidf.pkl"
    )
    combined_features = download_file(
        "https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/combined_features.pkl",
        "combined_features.pkl"
    )
    drugs_csv_path = download_file(
        "https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/drugs.csv",
        "drugs.csv",
        is_pickle=False
    )
    drugs_df = pd.read_csv(drugs_csv_path)
    return grouped, similarity_matrix, tfidf, combined_features, drugs_df

# ------------------ 🔍 Recommendation Logic ------------------ #
def recommend(drug_name, grouped, combined_features, top_n=5):
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

# ------------------ 🔽 UI ------------------ #
st.set_page_config(page_title="Medicine Recommendation", layout="wide")
st.title("🧠 Personalized Medicine Recommendation System")

grouped, similarity_matrix, tfidf, combined_features, drugs_df = load_all()

drug_list = sorted(drugs_df['drugName'].dropna().unique())
drug_name = st.selectbox("🔽 Select a medicine:", drug_list)

if drug_name:
    with st.spinner("🔎 Finding similar medicines..."):
        output = recommend(drug_name, grouped, combined_features)
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
