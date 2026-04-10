import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
# 🔥 Custom UI Styling
st.markdown("""
<style>

/* Background Image with DARK overlay */
.stApp {
    background: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)),
                url("https://images.unsplash.com/photo-1504711434969-e33886168f5c");
    background-size: cover;
    background-attachment: fixed;
}

/* Transparent box effect */
.main {
    background-color: rgba(0, 0, 0, 0.6);
    padding: 20px;
    border-radius: 15px;
}

/* Title styling */
h1 {
    color: #FFD700;
    text-align: center;
    font-family: 'Trebuchet MS', sans-serif;
}

/* Text styling */
.stTextArea textarea {
    background-color: rgba(255,255,255,0.9);
    color: black;
    font-size: 16px;
}

/* Button styling */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px;
}

/* Success & Error styling */
.stSuccess {
    background-color: rgba(0, 255, 0, 0.2);
}
.stError {
    background-color: rgba(255, 0, 0, 0.2);
}

</style>
""", unsafe_allow_html=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download once
# nltk.download('stopwords')

# Initialize
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 🔹 Text Preprocessing
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)


# 🔹 Train model (cached → runs only once)
@st.cache_resource
def load_and_train():

    # Load dataset
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')

    # Labels
    fake['label'] = 1
    true['label'] = 0

    # Combine + shuffle
    df = pd.concat([fake, true], axis=0)
    df = df.sample(frac=1, random_state=2)

    # Use title + text
    df['content'] = df['title'] + " " + df['text']

    # Preprocess
    df['content'] = df['content'].apply(stemming)

    X = df['content']
    y = df['label']

    # TF-IDF (LIMIT FEATURES → faster)
    vector = TfidfVectorizer(max_features=3000)
    X = vector.fit_transform(X)

    # Model
    model = LogisticRegression(max_iter=500, class_weight='balanced')
    model.fit(X, y)

    return model, vector


# Load model (runs once only)
model, vector = load_and_train()


# 🔹 UI
st.markdown(
    "<h1 style='color:#FF5733; text-align:center; font-weight:bold;'>📰 Fake News Detection App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='color:#00FFFF; text-align:center; font-weight:bold;'>Check whether a news article is Real or Fake</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:#FFD700; font-size:18px; font-weight:bold;'>Enter News Text:</p>",
    unsafe_allow_html=True
)

input_text = st.text_area("")

# 🔹 Prediction
def predict(text):
    processed = stemming(text)
    vec = vector.transform([processed])

    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)

    confidence = np.max(proba) * 100
    return pred, confidence


# 🔹 Button
if st.button("Check News"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        pred, confidence = predict(input_text)

        if pred == 1:
            st.error(f"🚨 Fake News\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ Real News\nConfidence: {confidence:.2f}%")