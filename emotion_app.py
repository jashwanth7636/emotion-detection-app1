import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import difflib
import requests

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Emotion Detection", layout="centered")

# ===============================
# UI DESIGN
# ===============================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 30px;
}
.stTextInput > div > div > input {
    border-radius: 10px;
    padding: 12px;
}
.stButton>button {
    border-radius: 10px;
    height: 45px;
    width: 150px;
    background-color: #4CAF50;
    color: white;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 20px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA (ONLINE)
# ===============================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"
    data = requests.get(url).json()
    texts = [d['content'] for d in data][:20000]
    return pd.DataFrame(texts, columns=["text"])

df = load_data()

# ===============================
# EMOTION LABELING
# ===============================
def label_emotion(text):
    text = text.lower()
    
    happy_words = ["love", "happy", "great", "good", "awesome", "fantastic", "glad"]
    sad_words = ["sorry", "sad", "bad", "cry", "pain", "upset", "hurt"]
    angry_words = ["kill", "hate", "angry", "mad", "furious"]

    score = {"happy": 0, "sad": 0, "angry": 0}

    for word in happy_words:
        if word in text:
            score["happy"] += 1
    for word in sad_words:
        if word in text:
            score["sad"] += 1
    for word in angry_words:
        if word in text:
            score["angry"] += 1

    if max(score.values()) == 0:
        return "neutral"

    return max(score, key=score.get)

df["emotion"] = df["text"].apply(label_emotion)

# ===============================
# CLEAN TEXT
# ===============================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# ===============================
# TRAIN MODEL
# ===============================
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["emotion"]

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = train_model()

# ===============================
# SPELL CORRECTION
# ===============================
def correct_spelling(word, vocabulary):
    matches = difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.8)
    return matches[0] if matches else word

# ===============================
# PREDICTION
# ===============================
def predict_emotion(text):
    text = clean_text(text)

    vocab = vectorizer.get_feature_names_out()
    words = text.split()
    corrected_words = [correct_spelling(w, vocab) for w in words]
    text = " ".join(corrected_words)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0]) * 100

    return prediction, confidence

# ===============================
# UI
# ===============================
st.markdown('<div class="big-title">🎬 Emotion Detection from Dialogues</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type your sentence and discover the emotion instantly</div>', unsafe_allow_html=True)

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input:
        result, confidence = predict_emotion(user_input)

        if result == "happy":
            color = "#2ecc71"
            emoji = "😊"
        elif result == "sad":
            color = "#3498db"
            emoji = "😢"
        elif result == "angry":
            color = "#e74c3c"
            emoji = "😡"
        else:
            color = "#95a5a6"
            emoji = "😐"

        st.markdown(f"""
        <div class="result-box" style="background-color:{color}; color:white;">
            {emoji} Emotion: {result} <br>
            Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Please enter some text")
