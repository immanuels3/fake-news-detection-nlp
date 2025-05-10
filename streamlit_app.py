# app.py
import gradio as gr
import re
import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set NLTK data path and download resources
nltk.data.path.append('/tmp/nltk_data')
try:
    nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'], download_dir='/tmp/nltk_data', quiet=True)
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

# Load transformer model
def load_transformer_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        transformer = pipeline("text-classification", model=model_name, device=-1)  # CPU
        logger.info("Transformer model loaded successfully")
        return transformer
    except Exception as e:
        logger.error(f"Error loading transformer model: {e}")
        raise Exception(f"Error loading transformer model: {e}")

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
        return error_msg, None, None

    # Preprocess text
    cleaned_text = preprocess_text(news_text)
    
    # Initialize outputs
    results = []
    word_cloud_img = None

    # Load transformer model
    try:
        transformer_model = load_transformer_model()
    except Exception as e:
        return str(e), None, None

    # Transformer model prediction
    result = transformer_model(news_text[:512])[0]
    prediction = "Fake" if result['label'] == "NEGATIVE" else "Real"
    confidence = result['score']
    results.append(f"**Transformer Model (DistilBERT) Prediction**: {prediction}")
    results.append(f"**Confidence**: {confidence:.2%}")

    # Word cloud
    word_cloud_fig = plot_word_cloud(cleaned_text)
    word_cloud_img = fig_to_base64(word_cloud_fig)
    plt.close(word_cloud_fig)

    return "\n".join(results), cleaned_text, word_cloud_img

# Gradio interface
def main():
    iface = gr.Interface(
        fn=analyze_news,
        inputs=gr.Textbox(lines=10, placeholder="Paste the news article here...", label="News Article Text"),
        outputs=[
            gr.Markdown(label="Prediction Results"),
            gr.Textbox(label="Processed Text"),
            gr.Image(label="Word Cloud")
        ],
        title="🔍 IndiaTruthGuard: Indian Fake News Detection",
        description="Enter a news article to detect if it's **Real** or **Fake**. Powered by DistilBERT.",
        theme="default"
    )
    iface.launch()

if __name__ == '__main__':
    main()
