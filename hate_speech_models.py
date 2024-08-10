from transformers import pipeline # We are using transformers for our pre-trained model

# Loading a pre-trained hate speech detection model
hate_speech_detector = pipeline("text-classification", model="unitary/toxic-bert")

# For this we are using the unitary/toxic-bert model from huggingface. Below is the page referencing/hosting this model:
# Source: https://huggingface.co/unitary/toxic-bert


# Good Model (Should pass CI/CD)
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
