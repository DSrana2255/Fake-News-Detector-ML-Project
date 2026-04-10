📰 Fake News Detection System (Streamlit + Machine Learning)
🚀 Project Overview

The Fake News Detection System is a machine learning web application built using Streamlit that classifies news articles as Real or Fake. It uses Natural Language Processing (NLP) techniques such as text preprocessing, TF-IDF vectorization, and Logistic Regression to analyze and predict news authenticity.

This project helps users identify misleading information and promotes awareness of fake news.

✨ Features
📰 Detects whether news is Real or Fake
🤖 Machine Learning model (Logistic Regression)
🧠 NLP-based text preprocessing (stemming, stopword removal)
📊 TF-IDF feature extraction
⚡ Fast real-time predictions
🎯 Confidence score for predictions
🎨 Beautiful UI using Streamlit custom CSS
📱 Simple and interactive web interface
🛠️ Tech Stack
Frontend/UI: Streamlit
Backend: Python
ML Algorithm: Logistic Regression
NLP Libraries: NLTK, re
Feature Extraction: TF-IDF Vectorizer
Dataset: Fake.csv and True.csv
📂 Project Structure
Fake-News-Detection/
│
├── app.py (Streamlit app)
├── Fake.csv
├── True.csv
├── requirements.txt (optional)
└── README.md
⚙️ How It Works
1️⃣ Data Collection
Dataset contains:
Fake.csv → Fake news articles
True.csv → Real news articles
2️⃣ Data Preprocessing

The text is cleaned using NLP techniques:

Removing special characters
Converting text to lowercase
Removing stopwords
Applying Porter Stemming
3️⃣ Feature Extraction
TF-IDF Vectorizer converts text into numerical format
Top 3000 features are selected for efficiency
4️⃣ Model Training
Logistic Regression model is trained on labeled data
Balanced class weights improve accuracy
5️⃣ Prediction Process
User enters news text
Text is preprocessed
Converted using TF-IDF
Model predicts:
Fake (1)
Real (0)
Confidence score is displayed
🧠 Machine Learning Workflow
Input Text →
Text Cleaning (Regex + NLP) →
Stemming + Stopword Removal →
TF-IDF Vectorization →
Logistic Regression Model →
Output Prediction (Fake / Real)
▶️ Installation & Setup
1. Clone Repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install Dependencies
pip install streamlit numpy pandas nltk scikit-learn
4. Download NLTK Data (First Time Only)
import nltk
nltk.download('stopwords')
5. Run the Application
streamlit run app.py
6. Open in Browser
http://localhost:8501
📊 Model Performance
Algorithm: Logistic Regression
Feature Type: TF-IDF
Accuracy: High performance on balanced dataset
Output: Binary classification (Real / Fake)


🔮 Future Improvements
🤖 Upgrade to Deep Learning (LSTM / BERT)
🌐 Add live news scraping (Google News API)
📊 Dashboard for analytics
📱 Mobile-friendly version
🧾 Explainable AI (why news is fake/real)
👨‍💻 Author

Deepak Kumar Rana
🎓 Final Year Project – Fake News Detection System
🛠 Built using Streamlit + NLP + Machine Learning

⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
