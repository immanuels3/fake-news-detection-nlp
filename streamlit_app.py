import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'], quiet=True)

# Enhanced text preprocessing
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
@st.cache_data
def load_dataset(fake_path='data/Fake.csv', true_path='data/True.csv', sample_size=1000):
    try:
        df_fake = pd.read_csv(fake_path, encoding='utf-8')
        df_true = pd.read_csv(true_path, encoding='utf-8')

        df_fake['label'] = 1
        df_true['label'] = 0

        if sample_size:
            df_fake = df_fake.sample(n=min(sample_size//2, len(df_fake)), random_state=42)
            df_true = df_true.sample(n=min(sample_size//2, len(df_true)), random_state=42)

        required_columns = ['title', 'text', 'subject', 'date']
        for df in [df_fake, df_true]:
            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV files must contain these columns: {required_columns}")
                return None

        df = pd.concat([df_fake, df_true], ignore_index=True)

        if not df['label'].isin([0,1]).all():
            logger.error("Labels must be binary (0 or 1)")
            return None

        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')

        df['combined_text'] = df['title'] + ' ' + df['text']
        df['cleaned_text'] = df['combined_text'].apply(preprocess_text)

        logger.info(f"Dataset loaded: {len(df)} articles")
        return df
    except FileNotFoundError as e:
        logger.error(f"One or both dataset files not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

# Load all models
@st.cache_resource
def load_models(path='models/'):
    try:
        vectorizer = joblib.load(f'{path}vectorizer.pkl')
        models = {
            'Logistic Regression': joblib.load(f'{path}Logistic_Regression.pkl'),
            'Random Forest': joblib.load(f'{path}Random_Forest.pkl'),
            'XGBoost': joblib.load(f'{path}XGBoost.pkl'),
            'Ensemble': joblib.load(f'{path}Ensemble.pkl')
        }
        transformer_model = pipeline("text-classification", 
                                  model="distilbert-base-uncased-finetuned-sst-2-english")
        logger.info("All models loaded successfully")
        return models, vectorizer, transformer_model
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Explain predictions
def explain_prediction(text, model, vectorizer):
    explainer = LimeTextExplainer(class_names=["True", "Fake"])
    def predict_prob(texts):
        return model.predict_proba(vectorizer.transform(texts))
    exp = explainer.explain_instance(text, predict_prob, num_features=10)
    return exp.as_list()

# Visualization functions
def plot_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

# Main app
def main():
    st.set_page_config(page_title="TruthGuard: Fake News Detection", 
                      page_icon="🔍", 
                      layout="wide")

    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0055aa;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/news.png")
        st.title("TruthGuard")
        st.markdown("Advanced Fake News Detection System")
        st.markdown("---")
        st.info("Enter a news article to analyze or view dataset visualizations.")

    # Main content
    st.title("🔍 TruthGuard: Fake News Detection")
    st.markdown("Analyze news articles to detect potential fake news using advanced NLP techniques.")

    # Tabs for different sections
    tab1, tab2 = st.tabs(["Analyze News", "Data Visualizations"])

    with tab1:
        st.subheader("Enter News Article")
        news_text = st.text_area("Paste your news article here:", height=200)
        
        if st.button("Analyze Article"):
            if news_text:
                is_valid, error_msg = validate_input_text(news_text)
                if not is_valid:
                    st.error(error_msg)
                else:
                    cleaned_text = preprocess_text(news_text)
                    
                    # Load models
                    models, vectorizer, transformer_model = load_models()
                    
                    if models and vectorizer and transformer_model:
                        # Transformer prediction
                        with st.spinner("Analyzing with Transformer model..."):
                            result = transformer_model(news_text[:512])[0]
                            prediction = "Fake" if result['label'] == "NEGATIVE" else "True"
                            confidence = result['score']
                        
                        st.markdown("### Transformer Model Results")
                        st.markdown(f"<div class='prediction-box'>"
                                  f"<b>Prediction:</b> {prediction}<br>"
                                  f"<b>Confidence:</b> {confidence:.2%}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        # Traditional models prediction
                        st.markdown("### Traditional Models Comparison")
                        vectorized_text = vectorizer.transform([cleaned_text])
                        
                        # Store results for table
                        results = []
                        for model_name, model in models.items():
                            with st.spinner(f"Analyzing with {model_name}..."):
                                prediction = model.predict(vectorized_text)[0]
                                proba = model.predict_proba(vectorized_text)[0]
                                results.append({
                                    'Model': model_name,
                                    'Prediction': 'Fake' if prediction == 1 else 'True',
                                    'Probability (True)': f"{proba[0]:.2%}",
                                    'Probability (Fake)': f"{proba[1]:.2%}"
                                })
                        
                        # Display results table
                        st.write("#### Prediction Summary")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df.style.set_properties(**{'text-align': 'center'}))
                        
                        # Detailed results for each model
                        for model_name, model in models.items():
                            st.markdown(f"#### {model_name} Results")
                            prediction = model.predict(vectorized_text)[0]
                            proba = model.predict_proba(vectorized_text)[0]
                            st.markdown(f"<div class='prediction-box'>"
                                      f"<b>Prediction:</b> {'Fake' if prediction == 1 else 'True'}<br>"
                                      f"<b>Probability (True):</b> {proba[0]:.2%}<br>"
                                      f"<b>Probability (Fake):</b> {proba[1]:.2%}"
                                      f"</div>", unsafe_allow_html=True)
                            
                            # Explain predictions
                            st.markdown(f"##### Feature Importance ({model_name})")
                            with st.spinner(f"Generating explanation for {model_name}..."):
                                explanation = explain_prediction(cleaned_text, model, vectorizer)
                                exp_df = pd.DataFrame(explanation, columns=['Feature', 'Weight'])
                                st.dataframe(exp_df.style.format({'Weight': '{:.3f}'}))
                        
                        # Word cloud
                        st.markdown("### Word Cloud")
                        fig = plot_word_cloud(cleaned_text)
                        st.pyplot(fig)
                    else:
                        st.error("Error: Could not load models. Please ensure all model files (Logistic_Regression.pkl, Random_Forest.pkl, XGBoost.pkl, Ensemble.pkl, vectorizer.pkl) are in the 'models/' directory.")
            else:
                st.warning("Please enter a news article to analyze.")

    with tab2:
        st.subheader("Data Visualizations")
        st.info("Visualizations generated from the provided dataset.")
        
        # Load dataset
        df = load_dataset()
        
        if df is not None:
            st.write("Dataset Preview")
            st.dataframe(df.head())
            
            if 'subject' in df.columns:
                st.write("Subject Distribution")
                fig, ax = plt.subplots(figsize=(10, 4))
                df['subject'].value_counts().plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    if not df['date'].isna().all():
                        st.write("Articles Over Time")
                        time_df = df.set_index('date').resample('ME').size()
                        fig, ax = plt.subplots(figsize=(10, 4))
                        time_df.plot(ax=ax)
                        st.pyplot(fig)
                except:
                    st.warning("Could not parse date column")
        else:
            st.error("Error: Could not load dataset. Please ensure 'Fake.csv' and 'True.csv' are in the 'data/' directory.")

if __name__ == '__main__':
    main()
