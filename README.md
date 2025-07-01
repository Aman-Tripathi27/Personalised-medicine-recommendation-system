# 🧠 Personalized Medicine Recommendation System  
[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-00c853?style=for-the-badge&logo=streamlit)](https://personalised-medicine-recommendation-system.streamlit.app/)

A data-driven solution for intelligent drug recommendations using machine learning, patient insights, and clinical knowledge.

---

## 🚀 Overview

The **Personalized Medicine Recommendation System** is a cutting-edge healthcare tool that analyzes real-world patient reviews, medication ratings, and clinical treatment data to recommend alternative medicines tailored to individual needs. This helps:

- 🧬 Improve treatment outcomes  
- ❌ Reduce adverse drug reactions  
- ⏱ Save time for healthcare professionals  
- 📈 Empower data-driven clinical decision-making

Built with ❤️ for modern healthcare systems and precision medicine enthusiasts.

---

## 🔍 Demo

👉 **[Launch the app live →](https://personalised-medicine-recommendation-system.streamlit.app/)**  
🧪 Enter a medicine name and instantly get 5 similar alternatives, complete with:
- Drug names  
- Treated conditions  
- Patient ratings  
- Clinical reviews  
- Cosine similarity scores (semantic similarity)

---

## 🧩 How It Works

1. **User Inputs Drug Name**  
2. **System Vectorizes Medical Features**  
3. **Computes Cosine Similarity**  
4. **Recommends Closest Medicines**  
5. **Displays Ratings + Reviews**

---

## 🗺 Roadmap

| Phase | Description |
|-------|-------------|
| 📥 **Data Collection** | Gathered from real-world reviews, drug metadata, and ratings |
| 🧹 **Preprocessing** | Cleaned, tokenized, vectorized using TF-IDF |
| 🧠 **ML Model** | Built similarity matrix using cosine similarity on combined features |
| 🖥 **UI/UX** | Streamlit-based clean, responsive web app |
| 📊 **Monitoring** | Future plan: add feedback loop and usage analytics |

---

## 🛠 Tech Stack

| Area              | Tool / Library |
|-------------------|----------------|
| Frontend          | Streamlit      |
| Language          | Python 3.13    |
| ML/NLP            | Scikit-learn, TF-IDF |
| Data Handling     | Pandas, NumPy  |
| File Hosting      | Hugging Face Datasets |
| Deployment        | Streamlit Cloud |

---

## 📂 Dataset & Files

All model files are hosted on Hugging Face:  
- [`grouped.pkl`](https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/grouped.pkl)  
- [`similarity_matrix.pkl`](https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/similarity_matrix.pkl)  
- [`combined_features.pkl`](https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/combined_features.pkl)  
- [`tfidf.pkl`](https://huggingface.co/datasets/aman1527/personalised-medicine-files/resolve/main/tfidf.pkl)

---

## ✨ Features

- 🔎 Intelligent medicine similarity search  
- 💬 Patient-centric: based on real reviews  
- 📊 Explainable results with ratings & condition context  
- 🌐 Lightweight, browser-friendly interface  
- ☁ Hosted online & open-source

---

## 📌 Future Improvements

- Add condition-based filtering  
- Improve recommendations using deep learning  
- Track user feedback to fine-tune results  
- Deploy on Hugging Face Spaces / Docker

---

## 🤝 Let's Collaborate

If you're passionate about **healthcare AI**, **clinical NLP**, or **drug discovery**, feel free to fork the repo, contribute ideas, or connect with me!

---

## 📣 Author

**Aman Tripathi**  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) &nbsp;&nbsp;&nbsp;🐦 [Twitter / X](https://x.com/yourprofile)  
🛠 Proud builder of tech for health.

---

> _“Precision medicine isn’t the future. It’s now.”_

