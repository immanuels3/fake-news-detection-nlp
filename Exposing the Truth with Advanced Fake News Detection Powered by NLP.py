# Indian Fake News Detection System for Google Colab
# Uses news_dataset.csv with columns: label, text

# Install required libraries
!pip install pandas numpy regex joblib matplotlib seaborn wordcloud scikit-learn xgboost transformers nltk lime torch textblob

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, roc_curve, auc, roc_auc_score)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import nltk
from lime.lime_text import LimeTextExplainer
from google.colab import files
from IPython.display import display
%matplotlib inline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set NLTK data path and download resources
nltk.data.path.append('/content/nltk_data')
try:
    nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'], download_dir='/content/nltk_data', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")
    print(f"Error downloading NLTK resources: {e}")

# Enhanced text preprocessing for English (Indian context)
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

# Data preprocessing function
def preprocess_data(df):
    print("\nData Preprocessing:")
    
    # Check for missing values
    print("Missing Values:")
    print(df.isnull().sum())
    df['text'] = df['text'].fillna('')
    
    # Check for duplicates
    print("\nDuplicate Rows:", df.duplicated().sum())
    df = df.drop_duplicates()
    
    # Add text length features
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    df['cleaned_text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    
    # Detect outliers in text length using IQR
    Q1 = df['text_length'].quantile(0.25)
    Q3 = df['text_length'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['text_length'] < Q1 - 1.5 * IQR) | (df['text_length'] > Q3 + 1.5 * IQR)]
    print("\nOutliers in Text Length:", len(outliers))
    
    # Visualize text length distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df['text_length'], kde=True, color='blue', label='Raw Text')
    sns.histplot(df['cleaned_text_length'], kde=True, color='orange', label='Cleaned Text')
    plt.title('Text Length Distribution (Before vs. After Preprocessing)')
    plt.xlabel('Word Count')
    plt.legend()
    plt.show()
    
    return df

# Feature engineering function
def feature_engineering(df):
    print("\nFeature Engineering:")
    
    # Add sentiment score
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    print("Sentiment Score Statistics:")
    print(df['sentiment'].describe())
    
    # Visualize sentiment distribution by label
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='label', y='sentiment', data=df)
    plt.title('Sentiment Score by Label (0=Real, 1=Fake)')
    plt.xlabel('Label')
    plt.ylabel('Sentiment Score')
    plt.show()
    
    # TF-IDF feature selection (top features)
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
    feature_names = vectorizer.get_feature_names_out()
    print("\nTop 10 TF-IDF Features:")
    print(feature_names[:10])
    
    return df

# Statistical analysis (EDA)
def statistical_analysis(df):
    print("\nStatistical Analysis (EDA):")
    
    # Univariate Analysis
    print("Class Distribution:")
    print(df['label'].value_counts())
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title('Class Distribution (0=Real, 1=Fake)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()
    
    # Word count analysis
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='text_length', hue='label', multiple='stack')
    plt.title('Text Length Distribution by Label')
    plt.xlabel('Word Count')
    plt.show()
    
    # Word clouds for real vs. fake news
    real_text = ' '.join(df[df['label'] == 0]['cleaned_text'])
    fake_text = ' '.join(df[df['label'] == 1]['cleaned_text'])
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(real_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud: Real News')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud: Fake News')
    plt.axis('off')
    plt.show()

# Load and prepare dataset from news_dataset.csv
def load_dataset(dataset_path, sample_size=1000):
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
        df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        required_columns = ['label', 'text']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV file must contain these columns: {required_columns}")
            return None
        if not df['label'].isin([0,1]).all():
            print("Error: Labels must be binary (0 for REAL, 1 for FAKE)")
            return None
        df['text'] = df['text'].fillna('')
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        logger.info(f"Dataset loaded: {len(df)} articles")
        return df
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Save models
def save_models(models, vectorizer, path='models/'):
    try:
        os.makedirs(path, exist_ok=True)
        joblib.dump(vectorizer, f'{path}vectorizer.pkl')
        for name, data in models.items():
            joblib.dump(data['model'], f'{path}{name.replace(" ", "_")}.pkl')
        logger.info("Models saved successfully")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        print(f"Error saving models: {e}")

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
            print("Error: No pre-trained models found in the specified directory.")
            return None, None
        logger.info("Models loaded successfully")
        return models, vectorizer
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        print(f"Error loading models: {e}")
        return None, None

# Train and evaluate multiple models
def train_models(df):
    if df is None:
        return None, None, None
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Ensemble": VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100)),
                ('xgb', XGBClassifier(use_label_encoder=False))
            ],
            voting='soft'
        )
    }
    results = {}
    best_model = None
    best_score = 0
    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name} ({i+1}/{len(models)})...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        y_proba = model.predict_proba(X_test_tfidf)[:, 1]
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='f1')
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        }
        results[name] = {
            'model': model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        }
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_model = name
        logger.info(f"Trained {name}: F1 Score = {metrics['f1']:.3f}")
    return results, vectorizer, best_model

# Load pre-trained transformer model for English
def load_transformer_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("Transformer model loaded successfully")
        return pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        logger.error(f"Error loading transformer model: {e}")
        print(f"Error loading transformer model: {e}")
        return None

# Explain model predictions using LIME
def explain_prediction(text, model, vectorizer):
    explainer = LimeTextExplainer(class_names=["Real", "Fake"])
    def predict_proba(texts):
        return model.predict_proba(vectorizer.transform(texts))
    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    return exp.as_list()

# Visualization functions
def plot_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Main function for Colab
def main():
    print("🔍 IndiaTruthGuard: Indian Fake News Detection")
    print("This system uses advanced NLP to detect fake news in Indian English news articles.")

    models = None
    vectorizer = None
    df = None
    transformer_model = load_transformer_model()
    best_model = None

    print("\nStep 1: Upload news_dataset.csv")
    uploaded_files = files.upload()

    dataset_path = None
    for filename in uploaded_files.keys():
        if 'news_dataset.csv' in filename:
            dataset_path = filename

    if dataset_path:
        df = load_dataset(dataset_path, sample_size=1000)
        if df is not None:
            print(f"Dataset loaded successfully: {len(df)} articles")
            print(f"Columns: {', '.join(df.columns)}")
            
            # Perform preprocessing, feature engineering, and statistical analysis
            df = preprocess_data(df)
            df = feature_engineering(df)
            statistical_analysis(df)
            
            print("Class Distribution:")
            print(df['label'].value_counts())
    else:
        print("Error: Please upload news_dataset.csv")
        return

    action = input("\nStep 2: Do you want to (1) Train new models or (2) Load pre-trained models? Enter 1 or 2: ")

    if action == '1':
        if df is not None:
            print("\nTraining models...")
            results, vectorizer, best_model = train_models(df)
            if results:
                models = results
                save_models(models, vectorizer)
                print("\nModels trained and saved successfully!")

                print("\nModel Performance:")
                metrics_df = pd.DataFrame({
                    model: data['metrics'] for model, data in results.items()
                }).T
                display(metrics_df)
                print(f"Best model: {best_model}")

                for name, data in models.items():
                    print(f"\nVisualizations for {name}:")
                    plot_confusion_matrix(data['y_test'], data['y_pred'])
                    plot_roc_curve(data['y_test'], data['y_proba'])
        else:
            print("Error: Dataset not loaded. Cannot train models.")

    elif action == '2':
        models, vectorizer = load_models()
        if models and vectorizer:
            best_model = max(models.items(), key=lambda x: x[1].get('metrics', {}).get('f1', 0))[0]
            print("Pre-trained models loaded successfully!")
        else:
            print("Error: Could not load pre-trained models.")

    print("\nStep 3: Analyze a news article")
    news_text = input("Enter the news article text (or paste it here): ")

    if news_text.strip():
        is_valid, error_msg = validate_input_text(news_text)
        if not is_valid:
            print(f"Error: {error_msg}")
        else:
            print("\nAnalyzing text...")
            cleaned_text = preprocess_text(news_text)

            print("\nProcessed Text:")
            print(cleaned_text)

            if transformer_model:
                result = transformer_model(news_text[:512])[0]
                prediction = "Fake" if result['label'] == "NEGATIVE" else "Real"
                confidence = result['score']

                print("\nTransformer Model Results:")
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.2%}")

                plot_word_cloud(cleaned_text, "Word Cloud of Processed Text")

            if models and vectorizer:
                print("\nTraditional Model Comparison:")
                vectorized_text = vectorizer.transform([cleaned_text])
                best_model_obj = models[best_model]['model']
                trad_prediction = best_model_obj.predict(vectorized_text)[0]
                trad_proba = best_model_obj.predict_proba(vectorized_text)[0]

                print(f"Best Model ({best_model}) Prediction: {'Fake' if trad_prediction == 1 else 'Real'}")
                print(f"Probability (Real): {trad_proba[0]:.2%}")
                print(f"Probability (Fake): {trad_proba[1]:.2%}")

                print("\nExplanation (Traditional Model):")
                explanation = explain_prediction(cleaned_text, best_model_obj, vectorizer)
                print("Top features influencing this prediction:")
                for feature, weight in explanation:
                    print(f"{feature}: {weight:.3f}")

    if df is not None:
        print("\nStep 4: Data Exploration")
        print("\nDataset Preview:")
        display(df.head())

if __name__ == '__main__':
    main()
