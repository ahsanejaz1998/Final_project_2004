import joblib
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the model
model = joblib.load('your_model.pkl')

# Load test data
tweet_df = pd.read_csv('test.csv')
X_test = tweet_df['tweet']
y_test = tweet_df['label']

# Transform data and predict
X_test_transformed = vect.transform(X_test)
predictions = model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Save evaluation result
with open("evaluation_result.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.2f}\n")
