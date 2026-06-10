import streamlit as st
import joblib
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved files
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = "".join(
        char for char in text
        if not char.isdigit()
    )

    text = "".join(
        char for char in text
        if char.isascii()
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

def predict_emotion(text):

    text = preprocess_text(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    emotion = label_encoder.inverse_transform(prediction)[0]

    return emotion

# ---------------- UI ---------------- #

st.set_page_config(
    page_title="Emotion Detection ",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Emotion Detection ")

st.markdown(
    "Analyze emotions from text using NLP and Machine Learning."
)

user_input = st.text_area(
    "Enter your text:",
    height=180
)

if st.button("Analyze Emotion"):

    if user_input.strip():

        emotion = predict_emotion(user_input)

        st.success(
            f"Predicted Emotion: {emotion}"
        )

    else:
        st.warning(
            "Please enter some text."
        )

st.markdown("""
<div class="blob blob1"></div>
<div class="blob blob2"></div>
<div class="blob blob3"></div>
""", unsafe_allow_html=True)

def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()