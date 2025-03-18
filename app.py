import streamlit as st
import pandas as pd
import spacy
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import os

# Ensure required dependencies are installed
subprocess.run(["pip", "install", "--upgrade", "pip"])
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Load Spacy Model (Assumes en_core_web_sm is installed via requirements.txt)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Spacy model 'en_core_web_sm' is missing. Ensure it's installed in requirements.txt.")
    st.stop()

# Function to Load Dataset with File Upload Option
@st.cache_data
def load_data():
    if os.path.exists("AI_Resume_Screening.csv"):
        return pd.read_csv("AI_Resume_Screening.csv")
    else:
        st.warning("Dataset not found! Please upload a CSV file.")
        uploaded_file = st.file_uploader("Upload AI_Resume_Screening.csv", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            return df
        return pd.DataFrame()

data = load_data()

# Function to Preprocess Text
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Job Matching Function
def match_resumes(job_desc, resumes):
    if not job_desc.strip():
        return []
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_desc] + resumes)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

# Streamlit UI
st.title("🚀 AI-Powered Resume Screening and Ranking System")

# Job Description Input
job_desc = st.text_area("Enter Job Description")

if st.button("Analyze Candidates"):
    if data.empty:
        st.warning("No data available for analysis. Please check your dataset.")
    else:
        data["processed_resume"] = (
            data["Skills"].fillna("") + " " + 
            data["Experience (Years)"].astype(str) + " years experience " + 
            data["Education"].fillna("")
        )
        data["processed_resume"] = data["processed_resume"].apply(preprocess_text)

        if job_desc.strip():
            data["match_score"] = match_resumes(preprocess_text(job_desc), data["processed_resume"].tolist())
            data["final_score"] = (data["match_score"] * 0.6) + (data["AI Score (0-100)"] / 100 * 0.4)

            ranked_data = data.sort_values(by="final_score", ascending=False)[
                ["Name", "Skills", "Experience (Years)", "Education", "AI Score (0-100)", "final_score"]
            ]

            # Display results
            st.subheader("🏆 Ranked Candidates")
            st.dataframe(ranked_data)

            # Visualization
            st.subheader("📊 Candidate Score Distribution")
            plt.figure(figsize=(8, 4))
            plt.hist(ranked_data["final_score"], bins=10, edgecolor="black")
            plt.xlabel("Final Score")
            plt.ylabel("Number of Candidates")
            plt.title("Distribution of Candidate Scores")
            st.pyplot(plt)
        else:
            st.warning("Please enter a job description before analyzing candidates.")
