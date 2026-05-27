from transformers import pipeline
from dataset_loader import load_imdb_dataset

from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
train_data, test_data = load_imdb_dataset()

# Shuffle Dataset
test_data = test_data.shuffle(seed=42)

# Small test sample
test_texts = test_data['text'][:1000]
test_labels = test_data['label'][:1000]

# Load Hugging Face Sentiment Pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

predictions = []

# Predict Sentiments
for text in test_texts:

    result = classifier(text[:512])[0]

    label = result['label']

    if label == "POSITIVE":
        predictions.append(1)
    else:
        predictions.append(0)

# Accuracy
accuracy = accuracy_score(test_labels, predictions)

print(f"\nTransformer Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(test_labels, predictions))