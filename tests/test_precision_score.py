# tests/test_precision_score.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score


def test_precision_score():
    # Load the dataset
    tweet_df = pd.read_csv('train.csv')

    # Preprocess the data
    tweet_df['tweet'] = tweet_df['tweet'].str.lower()
    tweet_df['tweet'] = tweet_df['tweet'].str.replace(
        r"https\S+|www\S+http\S+", '', regex=True)
    tweet_df['tweet'] = tweet_df['tweet'].str.replace(
        r'\@w+|\#', '', regex=True)
    tweet_df['tweet'] = tweet_df['tweet'].str.replace(
        r'[^\w\s]', '', regex=True)
    tweet_df['tweet'] = tweet_df['tweet'].str.replace(r'ð', '', regex=True)

    # Define the vectorizer and transform the data
    vect = joblib.load('vectorizer.pkl')
    X = vect.transform(tweet_df['tweet'])
    y = tweet_df['label']

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Load the model
    model = joblib.load('logreg_model.pkl')

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate precision score
    precision = precision_score(y_test, y_pred)

    # Set a threshold for the precision score
    threshold = 0.8
    if precision < threshold:
        raise ValueError(
            f"Model precision {precision:.2f} is below the threshold {threshold}")

    print("Precision score check passed.")


if __name__ == '__main__':
    test_precision_score()
