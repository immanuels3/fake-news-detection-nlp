# Indian Fake News Detection Web App using Gradio

import gradio as gr
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
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from lime.lime_text import LimeTextExplainer
import io
import base64

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
    raise Exception(f"Error downloading NLTK resources: {e}")

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

# Load models with enhanced error handling
def load_models(path='models/'):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory '{path}' does not exist.")
        vectorizer_path = f'{path}vectorizer.pkl'
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file '{vectorizer_path}' not found.")
        vectorizer = joblib.load(vectorizer_path)
        model_files = glob.glob(f'{path}*.pkl')
        models = {}
        for file in model_files:
            name = os.path.basename(file).replace('.pkl', '').replace('_', ' ')
            if name != 'vectorizer':
                try:
                    models[name] = {'model': joblib.load(file)}
                except Exception as e:
                    logger.error(f"Failed to load model {file}: {e}")
                    continue
        if not models:
            raise ValueError("No valid pre-trained models found in the specified directory.")
        logger.info("Models loaded successfully")
        return models, vectorizer
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise Exception(f"Error loading models: {str(e)}")

# Load transformer model
def load_transformer_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        transformer = pipeline("text-classification", model=model_name)
        logger.info("Transformer model loaded successfully")
        return transformer
    except Exception as e:
        logger.error(f"Error loading transformer model: {e}")
        raise Exception(f"Error loading transformer model: {e}")

# Explain predictions using LIME
def explain_prediction(text, model, vectorizer):
    explainer = LimeTextExplainer(class_names=["Real", "Fake"])
    def predict_proba(texts):
        return model.predict_proba(vectorizer.transform(texts))
    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    return exp.as_list()

# Convert matplotlib figure to base64 for Gradio
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{img_str}"

# Visualization function
def plot_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title("Word Cloud of Processed Text", fontsize=16)
    ax.axis('off')
    return fig

# Main prediction function for Gradio
def analyze_news(news_text):
    # Validate input
    is_valid, error_msg = validate_input_text(news_text)
    if not is_valid:
        return error_msg, None, None, None

    # Preprocess text
    cleaned_text = preprocess_text(news_text)
    
    # Initialize outputs
    results = []
    word_cloud_img = None
    lime_explanation = []

    # Load models
    try:
        models, vectorizer = load_models()
        transformer_model = load_transformer_model()
    except Exception as e:
        return str(e), None, None, None

    # Transformer model prediction
    if transformer_model:
        result = transformer_model(news_text[:512])[0]
        prediction = "Fake" if result['label'] == "NEGATIVE" else "Real"
        confidence = result['score']
        results.append(f"**Transformer Model (DistilBERT) Prediction**: {prediction}")
        results.append(f"**Confidence**: {confidence:.2%}")

    # Traditional model prediction
    if models and vectorizer:
        vectorized_text = vectorizer.transform([cleaned_text])
        best_model_name = max(models.items(), key=lambda x: x[1].get('metrics', {}).get('f1', 0))[0]
        best_model_obj = models[best_model_name]['model']
        trad_prediction = best_model_obj.predict(vectorized_text)[0]
        trad_proba = best_model_obj.predict_proba(vectorized_text)[0]

        results.append(f"**Best Traditional Model ({best_model_name}) Prediction**: {'Fake' if trad_prediction == 1 else 'Real'}")
        results.append(f"**Probability (Real)**: {trad_proba[0]:.2%}")
        results.append(f"**Probability (Fake)**: {trad_proba[1]:.2%}")

        # LIME explanation
        lime_explanation = explain_prediction(cleaned_text, best_model_obj, vectorizer)
        lime_explanation = [f"{feature}: {weight:.3f}" for feature, weight in lime_explanation]

    # Word cloud
    word_cloud_fig = plot_word_cloud(cleaned_text)
    word_cloud_img = fig_to_base64(word_cloud_fig)
    plt.close(word_cloud_fig)

    return "\n".join(results), cleaned_text, word_cloud_img, lime_explanation

# Gradio interface
def main():
    iface = gr.Interface(
        fn=analyze_news,
        inputs=gr.Textbox(lines=10, placeholder="Paste the news article here...", label="News Article Text"),
        outputs=[
            gr.Markdown(label="Prediction Results"),
            gr.Textbox(label="Processed Text"),
            gr.Image(label="Word Cloud"),
            gr.Textbox(label="LIME Explanation (Top Features)")
        ],
        title="🔍 IndiaTruthGuard: Indian Fake News Detection",
        description="Enter a news article to detect if it's **Real** or **Fake**. Powered by advanced NLP models.",
        theme="default"
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == '__main__':
    main()
