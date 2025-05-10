# app.py
# Indian Fake News Detection Web App using Gradio with Dataset Support

import gradio as gr
import pandas as pd
import numpy as np
import re
import joblib
import os
import glob
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import torch
from torch.utils.data import Dataset as TorchDataset

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

# Load dataset
def load_dataset(dataset_path='news_dataset.csv'):
    try:
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file {dataset_path} not found. Skipping training.")
            return None
        df = pd.read_csv(dataset_path)
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")
        df['text'] = df['text'].apply(preprocess_text)
        df = df[df['text'].str.strip() != '']
        logger.info("Dataset loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

# Train traditional models
def train_models(df, path='models/'):
    try:
        if df is None:
            logger.warning("No dataset provided. Skipping training.")
            return None, None
        X, y = df['text'], df['label']
        vectorizer = TfidfVectorizer(max_features=5000)
        X_vec = vectorizer.fit_transform(X)
        
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }
        trained_models = {}
        for name, model in models.items():
            model.fit(X_vec, y)
            y_pred = model.predict(X_vec)
            f1 = f1_score(y, y_pred)
            trained_models[name] = {'model': model, 'metrics': {'f1': f1}}
            joblib.dump(model, f'{path}{name.replace(" ", "_")}.pkl')
            logger.info(f"Trained {name} with F1 score: {f1:.4f}")
        
        joblib.dump(vectorizer, f'{path}vectorizer.pkl')
        logger.info("Vectorizer and models saved successfully")
        return trained_models, vectorizer
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return None, None

# Custom dataset class for DistilBERT
class NewsDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Fixed: Removed invalid hyphen
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Fine-tune DistilBERT
def fine_tune_distilbert(df, model_path='fine_tuned_distilbert'):
    try:
        if df is None:
            logger.warning("No dataset provided. Skipping DistilBERT fine-tuning.")
            return None
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        dataset = NewsDataset(df['text'].tolist(), df['label'].tolist(), tokenizer)
        training_args = TrainingArguments(
            output_dir='./distilbert_fine_tune',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info("DistilBERT fine-tuned and saved successfully")
        return pipeline("text-classification", model=model_path, tokenizer=model_path)
    except Exception as e:
        logger.error(f"Error fine-tuning DistilBERT: {e}")
        return None

# Load models
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
            raise Exception("No pre-trained models found in the specified directory.")
        logger.info("Models loaded successfully")
        return models, vectorizer
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None

# Load transformer model
def load_transformer_model(fine_tuned_path='fine_tuned_distilbert'):
    try:
        if os.path.exists(fine_tuned_path):
            transformer = pipeline("text-classification", model=fine_tuned_path, tokenizer=fine_tuned_path)
            logger.info("Fine-tuned DistilBERT model loaded successfully")
        else:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            transformer = pipeline("text-classification", model=model_name)
            logger.info("Default DistilBERT model loaded successfully")
        return transformer
    except Exception as e:
        logger.error(f"Error loading transformer model: {e}")
        return None

# Main prediction function for Gradio
def analyze_news(news_text):
    # Validate input
    is_valid, error_msg = validate_input_text(news_text)
    if not is_valid:
        return error_msg

    # Preprocess text
    cleaned_text = preprocess_text(news_text)
    
    # Load models
    try:
        models, vectorizer = load_models()
        transformer_model = load_transformer_model()
    except Exception as e:
        return str(e)

    # Use transformer model for prediction
    if transformer_model:
        result = transformer_model(news_text[:512])[0]
        prediction = "Fake" if result['label'] in ["NEGATIVE", "LABEL_1"] else "Real"
        return f"The news is predicted to be: **{prediction}**"
    
    # Fallback to traditional model
    if models and vectorizer:
        vectorized_text = vectorizer.transform([cleaned_text])
        best_model_name = max(models.items(), key=lambda x: x[1].get('metrics', {}).get('f1', 0))[0]
        best_model_obj = models[best_model_name]['model']
        trad_prediction = best_model_obj.predict(vectorized_text)[0]
        return f"The news is predicted to be: **{'Fake' if trad_prediction == 1 else 'Real'}**"

    return "Error: No valid model available for prediction."

# Main function
def main():
    # Load and train with dataset if available
    dataset_path = 'news_dataset.csv'
    df = load_dataset(dataset_path)
    if df is not None:
        # Train traditional models
        trained_models, vectorizer = train_models(df)
        # Fine-tune DistilBERT (optional, comment out if not needed)
        fine_tune_distilbert(df)
    
    # Launch Gradio interface
    iface = gr.Interface(
        fn=analyze_news,
        inputs=gr.Textbox(lines=10, placeholder="Paste the news article here...", label="News Article Text"),
        outputs=gr.Markdown(label="Prediction Result"),
        title="🔍 IndiaTruthGuard: Indian Fake News Detection",
        description="Enter a news article to detect if it's **Real** or **Fake**. Powered by advanced NLP models.",
        theme="default"
    )
    iface.launch()

if __name__ == '__main__':
    main()
