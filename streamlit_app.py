import streamlit as st
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources with error handling
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        st.info("NLTK resources downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        raise

download_nltk_resources()

# Custom Dataset class for PyTorch
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load and prepare dataset
def load_dataset():
    dataset_path = './data/news_dataset.csv'  # Adjust path as needed
    if not os.path.exists(dataset_path):
        st.error(f"Dataset file not found at {dataset_path}")
        return None

    # Load CSV
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

    # Verify columns
    if 'text' not in df.columns or 'label' not in df.columns:
        st.error("Dataset must contain 'text' and 'label' columns.")
        return None

    # Preprocess texts
    df['text'] = df['text'].apply(preprocess_text)
    
    # Map labels: FAKE=0, REAL=1
    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    df = df.dropna(subset=['text', 'label'])
    
    return df

# Compute metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Train and evaluate model
@st.cache_resource
def train_and_evaluate_model():
    # Load dataset
    df = load_dataset()
    if df is None:
        raise ValueError("Failed to load dataset.")
    
    # Split dataset: 80% train, 20% test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    train_texts = train_df['text'].values
    train_labels = train_df['label'].values
    test_texts = test_df['text'].values
    test_labels = test_df['label'].values

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Evaluate on test set
    eval_results = trainer.evaluate()
    st.write(f"**Test Set Evaluation Results:**")
    st.write(f"- Accuracy: {eval_results['eval_accuracy']:.4f}")
    st.write(f"- F1 Score: {eval_results['eval_f1']:.4f}")
    st.write(f"- Precision: {eval_results['eval_precision']:.4f}")
    st.write(f"- Recall: {eval_results['eval_recall']:.4f}")

    # Save model and tokenizer
    model.save_pretrained('./fake_news_model')
    tokenizer.save_pretrained('./fake_news_model')

    return model, tokenizer, eval_results

# Prediction function
def predict_fake_news(text, model, tokenizer):
    # Preprocess input text
    processed_text = preprocess_text(text)
    
    # Tokenize input
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return "Real News" if prediction == 1 else "Fake News"

# Streamlit app
def main():
    st.title("Indian Fake News Detector")
    st.markdown(
        """
        Enter a news article text below to check if it's real or fake.  
        Built for Indian news context using BERT and NLP with the `news_dataset.csv` dataset.  
        Trained on 80% of the dataset, evaluated on 20% test set.
        """
    )

    # Train model and cache results
    try:
        model, tokenizer, eval_results = train_and_evaluate_model()
    except Exception as e:
        st.error(f"Error training model: {e}")
        return

    # Input text
    user_input = st.text_area("Enter news article text:", placeholder="Paste the news article here...", height=200)

    # Predict button
    if st.button("Check News"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                prediction = predict_fake_news(user_input, model, tokenizer)
                st.success(f"Prediction: **{prediction}**")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
