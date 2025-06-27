# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from io import StringIO

# Load CSV from Google Drive
@st.cache_data
def load_csv_from_drive():
    file_id = "1shXUL3RkrSz_NDYsqFd2XU_Y3DNvMkOF"
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    data = StringIO(response.text)
    return pd.read_csv(data)

# Load models
@st.cache_data
def load_models():
    pt = pickle.load(open("pt.pkl", "rb"))
    similarity_scores = pickle.load(open("similarity_scores.pkl", "rb"))
    return pt, similarity_scores

# Load everything
med = load_csv_from_drive()
pt, similarity_scores = load_models()

# Recommender Function
def recommend(drug_name):
    if drug_name not in pt.index:
        return []

    index = np.where(pt.index == drug_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similar_items:
        item = []
        temp_df = med[med['drugName'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('drugName')['drugName'].values))
        item.extend(list(temp_df.drop_duplicates('drugName')['condition'].values))
        item.extend(list(temp_df.drop_duplicates('drugName')['review'].values))
        item.extend(list(temp_df.drop_duplicates('drugName')['rating'].values))
        data.append(item)
    return data

# Streamlit UI
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
