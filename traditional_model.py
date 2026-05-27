from dataset_loader import load_imdb_dataset
from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
train_data, test_data = load_imdb_dataset()

# Shuffle Dataset
train_data = train_data.shuffle(seed=42)
test_data = test_data.shuffle(seed=42)

# Select Samples
train_texts = [clean_text(text) for text in train_data['text'][:5000]]
train_labels = train_data['label'][:5000]

test_texts = [clean_text(text) for text in test_data['text'][:1000]]
test_labels = test_data['label'][:1000]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, train_labels)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(test_labels, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(test_labels, predictions))