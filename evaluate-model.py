import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#from hate_speech_models import hate_speech_detection  # Import the hate_speech_detection function

# Load the dataset
tweet_df = pd.read_csv('train.csv')

# Split the data into features and target
X = tweet_df['tweet']
Y = tweet_df['label']

# Convert target labels to a common format ('toxic' and 'non-toxic')
Y_binary = ['toxic' if label == 1 else 'non-toxic' for label in Y]

# Initialize lists to store predictions and true labels
predictions = []
true_labels = []

# Use the hate_speech_detection function to get predictions
for text, true_label in zip(X, Y_binary):
    filtered_text, censored_words = hate_speech_detection(text)
    # Assuming 'toxic' if there are censored words, 'non-toxic' otherwise
    prediction = 'toxic' if censored_words else 'non-toxic'
    predictions.append(prediction)
    true_labels.append(true_label)

# Evaluate the performance
accuracy = accuracy_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions, labels=['toxic', 'non-toxic'])
class_report = classification_report(true_labels, predictions, target_names=['toxic', 'non-toxic'])

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{class_report}")

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['toxic', 'non-toxic'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Save the evaluation results to a file
with open("evaluation_result.txt", "w") as file:
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Classification Report:\n{class_report}\n")
