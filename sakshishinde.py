# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:32:52 2024

@author: saksh
"""

import streamlit as st
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Tokenize, lemmatize, and remove stopwords
def tokenize_lemmatize_and_remove_stopwords(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    word_freq = defaultdict(int)

    for sentence in sentences:
        for word in word_tokenize(sentence):
            word_lemma = lemmatizer.lemmatize(word.lower())
            if word_lemma not in stop_words:
                word_freq[word_lemma] += 1

    return sentences, word_freq

# Step 3: Rank sentences using TF-IDF
def tfidf_summary(sentences, num_sentences=5):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    ranked_sentence_indices = sentence_scores.argsort()[-num_sentences:]
    ranked_sentences = [sentences[i] for i in ranked_sentence_indices]
    return " ".join(ranked_sentences)

# Step 4: Improve readability and coherence
def improve_summary(summary):
    sentences = sent_tokenize(summary)
    simplified_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        simplified_sentence = " ".join(words)
        simplified_sentences.append(simplified_sentence)
    
    improved_summary = " ".join(simplified_sentences)
    return improved_summary

# Streamlit app
st.title('Legal Document Summarization')
st.write("Upload a PDF document or enter text to generate a summary.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Text area
input_text = st.text_area("Or paste your text here")

# Process PDF or input text
if uploaded_file is not None or input_text:
    if uploaded_file is not None:
        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = input_text

    # Tokenize, lemmatize, remove stopwords, and calculate TF-IDF scores
    sentences, word_freq = tokenize_lemmatize_and_remove_stopwords(text)
    extractive_summary = tfidf_summary(sentences)

    # Improve the summary
    final_summary = improve_summary(extractive_summary)

    st.subheader("Extractive Summary")
    st.write(final_summary)
else:
    st.write("Please upload a PDF file or enter text to generate the summary.")
