import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load files
with open("med.pkl", "rb") as f:
    med = pickle.load(f)

with open("combined_features.pkl", "rb") as f:
    combined_features = pickle.load(f)

# Recommendation function
def recommend(drug_name, top_n=5):
    indices = med[med['drugName'].str.lower() == drug_name.lower()].index.tolist()
    if not indices:
        return []
    index = indices[0]
    sim_scores = cosine_similarity(combined_features[index], combined_features).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
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

# UI
st.title("💊 Personalized Medicine Recommender")
selected_drug = st.selectbox("Select a medicine", sorted(med['drugName'].unique()))

if st.button("Recommend"):
    output = recommend(selected_drug)
    if not output:
        st.error("No similar medicines found.")
    else:
        for o in output:
            st.markdown(f"""
            ---
            🧪 **Medicine**: `{o['🧪 Medicine']}`  
            📋 **Condition**: _{o['📋 Condition']}_  
            ⭐ **Rating**: {o['⭐ Rating']}  
            📊 **Similarity**: {o['📊 Similarity Score']}  
            🗣 **Review**: {o['🗣 Review']}
            """)
