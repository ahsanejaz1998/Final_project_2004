import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load the dataset
tweet_df = pd.read_csv('train.csv')

# Split the data into features and target
X = tweet_df['tweet']
Y = tweet_df['label']

# Vectorize the text data
vect = TfidfVectorizer()
X_vect = vect.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_vect, Y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
linreg = LinearRegression()
linreg.fit(X_train, Y_train)

# Predict using the Linear Regression model
Y_pred = linreg.predict(X_test)

# Convert continuous predictions to binary (0 or 1)
Y_pred_binary = [1 if pred >= 0.5 else 0 for pred in Y_pred]

# Evaluate the performance
linreg_acc = accuracy_score(Y_test, Y_pred_binary)
conf_matrix = confusion_matrix(Y_test, Y_pred_binary)
class_report = classification_report(Y_test, Y_pred_binary)


print(f"Accuracy: {linreg_acc}")


# Save the accuracy to a file
with open("evaluation_result.txt", "w") as file:
    file.write(f"Accuracy: {linreg_acc}\n")
