import streamlit as st
import pandas as pd
import spacy
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Ensure Spacy Model is Installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("AI_Resume_Screening.csv")  # Ensure dataset is in the same directory
    except FileNotFoundError:
        st.error("Dataset file 'AI_Resume_Screening.csv' not found. Please upload it.")
        return pd.DataFrame()

data = load_data()

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Job matching function
def match_resumes(job_desc, resumes):
    if not resumes:
        return []
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_desc] + resumes)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

# Stream
