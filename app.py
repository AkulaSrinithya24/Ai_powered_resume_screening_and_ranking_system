import streamlit as st
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("/content/drive/MyDrive/project1/AI_Resume_Screening.csv")

data = load_data()

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Job matching function
def match_resumes(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_desc] + resumes)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

# Streamlit UI
st.title("AI-Powered Resume Screening and Ranking System")

# Job Description Input
job_desc = st.text_area("Enter Job Description")

if st.button("Analyze Candidates"):
    data["processed_resume"] = data["Skills"].fillna("") + " " + data["Experience (Years)"].astype(str) + " years experience " + data["Education"].fillna("")
    data["processed_resume"] = data["processed_resume"].apply(preprocess_text)

    data["match_score"] = match_resumes(preprocess_text(job_desc), data["processed_resume"].tolist())
    data["final_score"] = (data["match_score"] * 0.6) + (data["AI Score (0-100)"] / 100 * 0.4)

    ranked_data = data.sort_values(by="final_score", ascending=False)[["Name", "Skills", "Experience (Years)", "Education", "AI Score (0-100)", "final_score"]]

    # Display results
    st.subheader("Ranked Candidates")
    st.dataframe(ranked_data)

    # Visualization
    st.subheader("Candidate Score Distribution")
    plt.figure(figsize=(8,4))
    plt.hist(ranked_data["final_score"], bins=10, edgecolor='black')
    plt.xlabel("Final Score")
    plt.ylabel("Number of Candidates")
    plt.title("Distribution of Candidate Scores")
    st.pyplot(plt)
