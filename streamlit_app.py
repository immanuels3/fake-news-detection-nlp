# app.py
# Indian Fake News Detection Web App using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from lime.lime_text import LimeTextExplainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set NLTK data path and download resources
nltk.data.path.append('./nltk_data')
try:
    nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'], download_dir='./nltk_data', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")
    st.error(f"Error downloading NLTK resources: {e}")

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    text = re.sub(r'\d+', 'NUMBER', text)
    try:
        tokens = word_tokenize(text.lower())
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        return ''
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens
             if token not in stopwords.words('english') and len(token) > 2]
    return ' '.join(tokens)

# Validate input text
def validate_input_text(text):
    if not text or len(text.strip()) < 10:
        return False, "Input text is too short or empty. Please provide a valid news article."
    return True, ""

# Load models
@st.cache_resource
def load_models(path='models/'):
    try:
        vectorizer = joblib.load(f'{path}vectorizer.pkl')
        model_files = glob.glob(f'{path}*.pkl')
        models = {}
        for file in model_files:
            name = os.path.basename(file).replace('.pkl', '').replace('_', ' ')
            if name != 'vectorizer':
                models[name] = {'model': joblib.load(file)}
        if not models:
            st.error("No pre-trained models found in the specified directory.")
            return None, None
        logger.info("Models loaded successfully")
        return models, vectorizer
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        return None, None

# Load transformer model
@st.cache_resource
def load_transformer_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        transformer = pipeline("text-classification", model=model_name)
        logger.info("Transformer model loaded successfully")
        return transformer
    except Exception as e:
        logger.error(f"Error loading transformer model: {e}")
        st.error(f"Error loading transformer model: {e}")
        return None

# Explain predictions using LIME
def explain_prediction(text, model, vectorizer):
    explainer = LimeTextExplainer(class_names=["Real", "Fake"])
    def predict_proba(texts):
        return model.predict_proba(vectorizer.transform(texts))
    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    return exp.as_list()

# Visualization functions
def plot_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title("Word Cloud of Processed Text", fontsize=16)
    ax.axis('off')
    return fig

# Streamlit UI
def main():
    st.set_page_config(page_title="IndiaTruthGuard", page_icon="🔍")
    st.title("🔍 IndiaTruthGuard: Indian Fake News Detection")
    st.markdown("Enter a news article to detect if it's **Real** or **Fake**. Powered by advanced NLP models.")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input("Model Directory", "models/")
    st.sidebar.info("Ensure pre-trained models and vectorizer are saved in the specified directory.")

    # Load models
    models, vectorizer = load_models(model_path)
    transformer_model = load_transformer_model()

    # Input text area
    news_text = st.text_area("Enter News Article Text", height=200, placeholder="Paste the news article here...")
    
    if st.button("Analyze"):
        if news_text.strip():
            is_valid, error_msg = validate_input_text(news_text)
            if not is_valid:
                st.error(error_msg)
            else:
                st.subheader("Analysis Results")
                cleaned_text = preprocess_text(news_text)

                # Display processed text
                with st.expander("Processed Text"):
                    st.write(cleaned_text)

                # Transformer model prediction
                if transformer_model:
                    result = transformer_model(news_text[:512])[0]
                    prediction = "Fake" if result['label'] == "NEGATIVE" else "Real"
                    confidence = result['score']

                    st.subheader("Transformer Model (DistilBERT) Results")
                    st.write(f"**Prediction**: {prediction}")
                    st.write(f"**Confidence**: {confidence:.2%}")

                    # Word cloud
                    st.subheader("Word Cloud")
                    fig = plot_word_cloud(cleaned_text)
                    st.pyplot(fig)

                # Traditional model prediction
                if models and vectorizer:
                    st.subheader("Traditional Model Results")
                    vectorized_text = vectorizer.transform([cleaned_text])
                    best_model_name = max(models.items(), key=lambda x: x[1].get('metrics', {}).get('f1', 0))[0]
                    best_model_obj = models[best_model_name]['model']
                    trad_prediction = best_model_obj.predict(vectorized_text)[0]
                    trad_proba = best_model_obj.predict_proba(vectorized_text)[0]

                    st.write(f"**Best Model ({best_model_name}) Prediction**: {'Fake' if trad_prediction == 1 else 'Real'}")
                    st.write(f"**Probability (Real)**: {trad_proba[0]:.2%}")
                    st.write(f"**Probability (Fake)**: {trad_proba[1]:.2%}")

                    # LIME explanation
                    st.subheader("Explanation (Traditional Model)")
                    explanation = explain_prediction(cleaned_text, best_model_obj, vectorizer)
                    st.write("**Top features influencing this prediction**:")
                    for feature, weight in explanation:
                        st.write(f"- {feature}: {weight:.3f}")
        else:
            st.warning("Please enter a news article to analyze.")

if __name__ == '__main__':
    main()
