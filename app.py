# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import bz2



# -----------------------------
# Load Data & Models
# -----------------------------
@st.cache_data
def load_all():
    med = pickle.load(bz2.BZ2File("med_compressed.pkl.bz2", "rb"))
    pt = pickle.load(open("pt.pkl", "rb"))
    similarity_scores = pickle.load(open("similarity_scores.pkl", "rb"))

    # Ensure similarity_scores is a NumPy array
    if isinstance(similarity_scores, pd.DataFrame):
        similarity_scores = similarity_scores.to_numpy()

    return med, pt, similarity_scores

med, pt, similarity_scores = load_all()

# -----------------------------
# Recommender Function
# -----------------------------
def recommend(drug_name):
    if drug_name not in pt.index:
        return []

    index = np.where(pt.index == drug_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:20]  # more to work with

    # Try filtering by same condition
    try:
        drug_condition = med[med['drugName'] == drug_name]['condition'].values[0]
    except IndexError:
        return []

    data = []
    for i in similar_items:
        temp_df = med[
            (med['drugName'] == pt.index[i[0]]) &
            (med['condition'] == drug_condition)
        ].drop_duplicates('drugName')

        if not temp_df.empty:
            item = [
                temp_df['drugName'].values[0],
                temp_df['condition'].values[0],
                temp_df['review'].values[0],
                temp_df['rating'].values[0]
            ]
            data.append(item)

        if len(data) == 5:
            break

    # Fallback: if nothing found with same condition
    if len(data) == 0:
        for i in similar_items:
            temp_df = med[med['drugName'] == pt.index[i[0]]].drop_duplicates('drugName')
            if not temp_df.empty:
                item = [
                    temp_df['drugName'].values[0],
                    temp_df['condition'].values[0],
                    temp_df['review'].values[0],
                    temp_df['rating'].values[0]
                ]
                data.append(item)
            if len(data) == 5:
                break

    return data



# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="💊 Medicine Recommender", layout="centered")
st.title("💊 Personalized Medicine Recommendation System")

# Dropdown for medicine selection
drug_list = sorted(pt.index.tolist())
selected_drug = st.selectbox("Select a medicine to find similar ones:", drug_list)

if st.button("Recommend"):
    results = recommend(selected_drug)

    if not results:
        st.error("No similar medicines found.")
    else:
        st.success("Top 5 Similar Medicines:")
        for item in results:
            st.markdown(f"**🧪 Medicine**: {item[0]}")
            st.markdown(f"**📋 Condition**: {item[1]}")
            st.markdown(f"**⭐ Rating**: {item[3]}")
            st.markdown(f"**🗣 Review**: {item[2][:300]}...")
            st.markdown("---")
