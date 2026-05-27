import streamlit as st

from transformers import pipeline

from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from dataset_loader import load_imdb_dataset

# Page Configuration
st.set_page_config(
    page_title="NLP vs Transformer",
    layout="centered"
)

# Title
st.title("NLP vs Transformer Comparison")

st.markdown(
    "Compare Traditional NLP and Hugging Face Transformer Models"
)

# Load Dataset
@st.cache_resource
def load_models():

    train_data, test_data = load_imdb_dataset()

    train_data = train_data.shuffle(seed=42)

    train_texts = [
        clean_text(text)
        for text in train_data['text'][:5000]
    ]

    train_labels = train_data['label'][:5000]

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train = vectorizer.fit_transform(train_texts)

    # Logistic Regression
    traditional_model = LogisticRegression(max_iter=1000)

    traditional_model.fit(X_train, train_labels)

    # Transformer Pipeline
    transformer_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return vectorizer, traditional_model, transformer_model


# Load Models
vectorizer, traditional_model, transformer_model = load_models()

# Input Box
user_input = st.text_area(
    "Enter a Movie Review",
    height=200,
    placeholder="Type your movie review here..."
)

# Analyze Button
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Traditional NLP Prediction
        cleaned_text = clean_text(user_input)

        vectorized_text = vectorizer.transform([cleaned_text])

        traditional_prediction = traditional_model.predict(
            vectorized_text
        )[0]

        traditional_result = (
            "Positive"
            if traditional_prediction == 1
            else "Negative"
        )

        # Transformer Prediction
        transformer_output = transformer_model(
            user_input,
            truncation=True,
            max_length=512
        )[0]

        transformer_result = transformer_output['label']

        confidence_score = transformer_output['score']

        # Results
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.success(
                f"Traditional NLP: {traditional_result}"
            )

        with col2:
            st.info(
                f"Transformer: {transformer_result}"
            )

        st.write(
            f"Transformer Confidence Score: "
            f"{confidence_score:.4f}"
        )