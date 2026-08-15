# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# DagsHub credentials for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = "rajeshxdatascience"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# MLflow tracking URI
repo_owner = "rajeshxdatascience"
repo_name = "yt-comment-sentiment-analysis"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)

app = FastAPI(title="YouTube Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CommentRequest(BaseModel):
    comments: list[str]

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):

    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed

@app.post("/predict")
def predict(request: CommentRequest):

    try:
        comments = request.comments

        print("Comments:", comments)

        # Preprocessing
        preprocessed_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        print("Preprocessed:", preprocessed_comments)

        # TF-IDF
        X = vectorizer.transform(preprocessed_comments)

        print("TF-IDF shape:", X.shape)

        # Prediction
        predictions = model.predict(X)

        print("Raw predictions:", predictions)


        sentiment_mapping = {
            1: "positive",
            0: "neutral",
            -1: "negative"
        }

        return [
            {
                "comment": comment,
                "sentiment": int(pred),
                "label": sentiment_mapping[int(pred)]
            }
            for comment, pred in zip(comments, predictions)
        ]

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "error": str(e),
            "type": type(e).__name__
        }

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=5000)