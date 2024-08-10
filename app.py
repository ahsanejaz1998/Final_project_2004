# Assignment #2: Movies Sentiment Analysis
# By: Justin Collier 
#     Ahsan Ejaz 
#     Jashanpreet Kaur Gill
# Date: 7/27/2024
# Purpose: Training four different models using the movies sentiment analysis dataset provided"

from flask import Flask, render_template, request

app = Flask(__name__)

# Importing our model function call
from hate_speech_models import hate_speech_detector

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# POST - Hate Speech Filtering Results
@app.route('/detect', methods=['POST'])
def detect():
    text = request.form['text']
    filtered_text, censored_words = hate_speech_detection(text)
    return render_template('index.html', original_text=text, filtered_text=filtered_text, censored_words=censored_words)

def hate_speech_detection(text):
    # Using the pre-trained model to detect hate speech
    results = hate_speech_detector(text)
    censored_words = []
    words = text.split()

    # If a word is flagged as toxic, it is censored, and its details (word and position) are logged.
    for i, word in enumerate(words):
        result = hate_speech_detector(word)
        if any(label['label'] == 'toxic' and label['score'] > 0.5 for label in result):
            censored_words.append((word, i + 1))
            words[i] = '****'
    
    filtered_text = ' '.join(words)
    return filtered_text, censored_words


# make the enumerate function available in the Jinja2 templates
@app.context_processor
def utility_processor():
    return dict(enumerate=enumerate)

if __name__ == '__main__':
    app.run(debug=True)
