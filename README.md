**IndiaTruthGuard: Indian Fake News Detection**

Detecting fake news in Indian English articles with advanced NLP techniques.
IndiaTruthGuard is a machine learning and natural language processing (NLP) powered system designed to classify Indian news articles as Real or Fake. The project addresses the growing challenge of misinformation in the Indian media landscape, providing tools for fact-checkers, journalists, and the public to combat false narratives. It features two main components:

Gradio Web App: A user-friendly interface deployed on Hugging Face Spaces, allowing users to input news text and receive instant predictions.
Google Colab Notebook: A research-oriented environment for data exploration, model training, evaluation, visualizations, and interpretability analysis.

The system leverages traditional machine learning models (Logistic Regression, Random Forest, XGBoost, VotingClassifier) and transformer-based models (DistilBERT) to achieve high accuracy, with a focus on Indian news contexts (e.g., articles related to events like the India-Pakistan conflict on May 10, 2025).

Table of Contents

Features
Project Workflow
Installation
Usage
Gradio Web App
Google Colab Notebook


Dataset
Models
Results
Contributing
Future Scope
License
Acknowledgements


Features

Fake News Detection: Classifies Indian English news articles as "Real" or "Fake" using advanced NLP.
Dual Interface:
Gradio App: Simple web interface for real-time predictions, deployed on Hugging Face Spaces.
Colab Notebook: Comprehensive research tool with data exploration, model training, and visualizations.


Robust Models:
Traditional ML: Logistic Regression, Random Forest, XGBoost, VotingClassifier.
Transformer: DistilBERT with optional fine-tuning.


Interpretability: LIME explanations for traditional model predictions (Colab version).
Visualizations: Word clouds, confusion matrices, ROC curves, and EDA plots (Colab version).
Indian Context: Tailored for Indian news, supporting datasets from sources like PIB Fact Check and AltNews.


Project Workflow
The project follows a structured workflow, visualized in the following Mermaid flowchart:
graph TD
    A[Start] --> B[Data Collection]
    B -->|Source news_dataset.csv| C[Data Preprocessing]
    C -->|Clean text, tokenize| D[Exploratory Data Analysis (EDA)]
    D -->|Analyze word frequency, class distribution| E[Feature Engineering]
    E -->|Create TF-IDF features, prepare for transformers| F{Train or Load Models?}
    F -->|Train| G[Model Building]
    G -->|Train Logistic Regression, Random Forest, XGBoost, VotingClassifier, Fine-tune DistilBERT| H[Model Evaluation]
    H -->|Assess accuracy, precision, recall, F1, ROC-AUC| I[Save Models]
    F -->|Load| J[Load Pre-trained Models]
    J --> K[Inference]
    I --> K
    K --> L{Deployment or Research?}
    L -->|Deployment| M[Gradio Web App]
    M -->|Launch on Hugging Face Spaces| N[User Inputs News Text]
    N --> O[Predict Real/Fake]
    L -->|Research| P[Google Colab Analysis]
    P -->|Visualizations, LIME Explanations, Metrics| Q[Generate Insights]
    O --> R[End]
    Q --> R

Stages:

Data Collection: Source news_dataset.csv with Indian news articles and labels.
Data Preprocessing: Clean text (remove URLs, emojis, punctuation) and tokenize.
EDA: Analyze word frequency, class distribution, and text patterns.
Feature Engineering: Create TF-IDF features and prepare text for transformers.
Model Building: Train traditional ML models and fine-tune DistilBERT.
Model Evaluation: Assess performance using accuracy, precision, recall, F1, and ROC-AUC.
Deployment: Launch Gradio app on Hugging Face Spaces.
Research Analysis: Use Colab for visualizations, LIME explanations, and metrics.


Installation
Prerequisites

Hardware: Minimum 4 GB RAM (8 GB recommended), optional GPU for DistilBERT fine-tuning.
Software: Python 3.10+, Google Colab (for research), Hugging Face Spaces (for deployment).

Dependencies
Install the required Python libraries using:
pip install pandas numpy regex joblib matplotlib seaborn wordcloud scikit-learn xgboost transformers nltk lime torch textblob gradio

For NLTK resources:
import nltk
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'], download_dir='./nltk_data')

Repository Setup

Clone the repository: git clone https://github.com/immanuels3/fake-news-detection-nlp
cd fake-news-detection-nlp


Place news_dataset.csv in the root directory (see Dataset).
Ensure models/ directory contains pre-trained models (optional for Gradio app).


Usage
Gradio Web App
The Gradio app provides a simple interface for real-time fake news detection.

Deploy on Hugging Face Spaces:

Create a new Space on Hugging Face.
Upload app.py, requirements.txt, news_dataset.csv (optional), and models/ directory.
Set requirements.txt:pandas
numpy
regex
joblib
scikit-learn
xgboost
transformers
nltk
lime
torch
gradio
textblob


Launch the Space to host the app.


Run Locally:
python app.py

Access the app at http://localhost:7860.

Usage:

Open the Gradio interface.
Paste a news article in the text box.
Click "Submit" to get a prediction ("Real" or "Fake").



Example:

Input: "India targeted Pakistan’s radar sites with minimal collateral damage."
Output: The news is predicted to be: Real

Google Colab Notebook
The Colab notebook is designed for research, model training, and analysis.

Setup:

Open colab_notebook.ipynb in Google Colab.
Upload news_dataset.csv when prompted.


Steps:

Data Preprocessing: Clean text, check missing values, duplicates, and outliers.
Feature Engineering: Add sentiment scores and TF-IDF features.
EDA: Visualize class distribution, text length, and word clouds.
Model Training: Train models or load pre-trained ones.
Evaluation: View metrics (accuracy, F1, ROC-AUC) and visualizations (confusion matrix, ROC curve).
Inference: Input a news article to predict "Real" or "Fake" with LIME explanations.


Run:

Execute all cells in Colab.
Follow prompts to upload the dataset and choose actions (train/load models, input news text).



Example Output:
Dataset loaded successfully: 1000 articles
Class Distribution:
0    500
1    500
[EDA Visualizations: Histograms, Word Clouds]
Best Model (Ensemble) Prediction: Fake
Probability (Real): 20.00%
Probability (Fake): 80.00%
Top features influencing prediction:
- attack: 0.152
- claim: 0.134


Dataset
The system uses a CSV dataset (news_dataset.csv) with Indian news articles.

Format:

Columns: text (news article/headline), label ("REAL" or "FAKE").
Labels are mapped to 0 (Real) and 1 (Fake) during preprocessing.


Sample:
text,label
"India targeted Pakistan’s radar sites with minimal collateral damage.",REAL
"Pakistan destroyed India’s S-400 system in drone attack.",FAKE
"Government announces free healthcare for all citizens.",REAL
"70% of India’s power grid fails due to cyberattack.",FAKE


Sources: Custom datasets or public sources like Kaggle, PIB Fact Check, AltNews.

Size: 1,000–10,000 articles (configurable via sample_size in Colab).


Note: Ensure the dataset is placed in the project root or uploaded in Colab.

Models
The system employs a combination of traditional ML and transformer models:

Traditional ML:

Logistic Regression: Fast, interpretable baseline.
Random Forest: Captures non-linear text patterns.
XGBoost: High-performance gradient boosting.
VotingClassifier: Ensemble of the above for improved accuracy.


Transformer:

DistilBERT (distilbert-base-uncased-finetuned-sst-2-english): Semantic text understanding, with optional fine-tuning.



Training Details:

Traditional ML: 80-20 train-test split, TF-IDF features (3000 max, n-grams), 5-fold cross-validation (Colab).
DistilBERT: Fine-tuned with 3 epochs (Gradio, optional).


Results
Performance (Colab):

VotingClassifier: F1 score ~85-90%, ROC-AUC ~0.9.
DistilBERT: Comparable F1 after fine-tuning.
Metrics: Accuracy, precision, recall, F1, ROC-AUC.

Visualizations (Colab):

Word clouds for real vs. fake news.
Confusion matrices and ROC curves for model evaluation.
Text length and sentiment distributions.

Gradio App:

Outputs simple "Real" or "Fake" predictions.
Prioritizes DistilBERT for inference, with fallback to traditional models.


Contributing
We welcome contributions to enhance IndiaTruthGuard! To contribute:

Fork the repository.
Create a branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Ideas:

Add multilingual support (e.g., Tamil news).
Integrate SHAP for transformer model explanations.
Expand the dataset with more Indian news sources.


Future Scope

Multilingual Support: Extend to regional languages (e.g., Hindi, Tamil) using mBERT.
Advanced Models: Implement BERT, RoBERTa, or neural networks.
Explainable AI: Add SHAP or LIME to the Gradio app for transparency.
Real-Time Detection: Develop an API for social media integration.
Collaboration: Partner with fact-checkers (e.g., AltNews, PIB) for real-world deployment.
Mobile App: Create a mobile interface for public access.


License
This project is licensed under the Creative Commons license. See the LICENSE file for details.

Acknowledgements

xAI: For providing Grok, which assisted in code development and documentation.
Hugging Face: For hosting the Gradio app and transformer models.
Google Colab: For providing a free research environment.


Learn and Share
