import joblib
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Load the stopwords
stop_words = set(stopwords.words('english'))

# Load the model and vectorizer
model = joblib.load('logreg_model.pkl')
vect = joblib.load('vectorizer.pkl')

# Load test data
tweet_df = pd.read_csv('train.csv')
X_test = tweet_df['tweet']
y_test = tweet_df['label']

# Process the data similarly as in training
def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub(r'ð','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)

tweet_df.tweet = tweet_df['tweet'].apply(data_processing)

# Transform test data using the loaded vectorizer
X_test_transformed = vect.transform(X_test)
predictions = model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Save evaluation result
with open("evaluation_result.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.2f}\n")
