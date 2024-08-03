# Assignment #2: Movies Sentiment Analysis
# By: Justin Collier
#     Ahsan Ejaz
#     Jashanpreet Kaur Gill
# Date: 7/27/2024
# Purpose: Training four different models using the movies sentiment analysis dataset provided"

from flask import Flask, render_template, request
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('logreg_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function (should match preprocessing in hate_speech.py)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+http\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'ð', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

# Home Page


@app.route('/')
def index():
    return render_template('index.html')

# POST - Hate Speech Filtering Results


@app.route('/detect', methods=['POST'])
def detect():
    text = request.form['text']
    filtered_text, censored_words = detect_hate_speech(text)
    return render_template('index.html', original_text=text, filtered_text=filtered_text, censored_words=censored_words)


def detect_hate_speech(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    # Transform the text using the vectorizer
    transformed_text = vectorizer.transform([preprocessed_text])
    # Predict hate speech (0 or 1) using the trained model
    prediction = model.predict(transformed_text)[0]
    # For demonstration, let's assume the model's output determines the hate speech words
    censored_words = []
    words = text.split()
    if prediction == 1:
        for i, word in enumerate(words):
            if word.lower() in ['hate', 'offensive', 'racist', 'threat']:
                censored_words.append((word, i + 1))
                words[i] = '****'
    filtered_text = ' '.join(words)
    return filtered_text, censored_words


@app.context_processor
def utility_processor():
    return dict(enumerate=enumerate)


if __name__ == '__main__':
    app.run(debug=True)
