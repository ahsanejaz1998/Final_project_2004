# Assignment #2: Movies Sentiment Analysis
# By: Justin Collier 
#     Ahsan Ejaz 
#     Jashanpreet Kaur Gill
# Date: 7/27/2024
# Purpose: Training four different models using the movies sentiment analysis dataset provided"

from flask import Flask, render_template, request

app = Flask(__name__)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# POST - Hate Speech Filtering Results
@app.route('/detect', methods=['POST'])
def detect():
    text = request.form['text']
    filtered_text = fake_hate_speech_detection(text)
    return render_template('index.html', original_text=text, filtered_text=filtered_text)

def fake_hate_speech_detection(text):

    # HARD CODED EXAMPLE - We need to put our classification model here to actually handle a library of hatespeech words.
    return text.replace('hate', '****').replace('offensive', '****').replace('racist', '****').replace('threat', '****')

if __name__ == '__main__':
    app.run(debug=True)
