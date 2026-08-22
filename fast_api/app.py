# app.py

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import mlflow
import joblib
import re
import pandas as pd
import matplotlib.dates as mdates

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from mlflow.tracking import MlflowClient

import uvicorn

import sys
import lightgbm
import sklearn


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(title="YouTube Sentiment Analysis API")


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


# ==========================================
# Request Models
# ==========================================

class CommentRequest(BaseModel):
    comments: list[str]


class CommentWithTimestamp(BaseModel):
    text: str
    timestamp: str


class CommentsWithTimestampRequest(BaseModel):
    comments: list[CommentWithTimestamp]


class SentimentCountsRequest(BaseModel):
    sentiment_counts: dict[str, int]


class SentimentDataRequest(BaseModel):
    sentiment_data: list[dict]


# ==========================================
# Preprocessing
# ==========================================

def preprocess_comment(comment):

    try:

        # Convert to lowercase
        comment = comment.lower()

        # Remove leading/trailing spaces
        comment = comment.strip()

        # Remove newline
        comment = re.sub(r"\n", " ", comment)

        # Remove non-alphanumeric characters
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]","",comment)

        # Stopwords
        stop_words = (set(stopwords.words("english"))- {"not", "but", "however", "no", "yet"})

        comment = " ".join([word for word in comment.split() if word not in stop_words])

        # Lemmatization
        lemmatizer = WordNetLemmatizer()

        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment

    except Exception as e:
        print(f"Error in preprocessing comment: {e}")

        return comment


# ==========================================
# Load Model + Vectorizer
# ==========================================

def load_model_and_vectorizer(model_name,model_version,vectorizer_path):

    mlflow.set_tracking_uri("https://dagshub.com/rajeshxdatascience/yt-comment-sentiment-analysis.mlflow")

    client = MlflowClient()

    model_uri = (f"models:/{model_name}/{model_version}")

    model = mlflow.pyfunc.load_model(model_uri)

    vectorizer = joblib.load(vectorizer_path)

    print("========== MODEL DEBUG ==========")
    print("Python:", sys.version)
    print("LightGBM:", lightgbm.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("Vectorizer vocabulary:", len(vectorizer.vocabulary_))
    print("=================================")  

    return model, vectorizer


# ==========================================
# Initialize Model
# ==========================================

model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model","2","./tfidf_vectorizer.pkl")


# ==========================================
# Home
# ==========================================

@app.get("/")
def home():

    return {"message":"Welcome to YouTube Sentiment Analysis API"}


# ==========================================
# Predict
# ==========================================

@app.post("/predict")
def predict(request: CommentRequest):

    comments = request.comments

    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        # Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # TF-IDF
        transformed_comments = (vectorizer.transform(preprocessed_comments))

        # Prediction
        predictions = model.predict(transformed_comments).tolist()

        # Convert to string
        predictions = [str(pred)for pred in predictions]

        X = vectorizer.transform(preprocessed_comments)

        print("Preprocessed:", preprocessed_comments)
        print("TF-IDF shape:", X.shape)
        print("TF-IDF non-zero:", X.nnz)

        predictions = model.predict(X)

        print("RAW PREDICTIONS:", predictions)

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Prediction failed: {str(e)}")

    return [{"comment": comment,"sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]


# ==========================================
# Predict With Timestamps
# ==========================================

@app.post("/predict_with_timestamps")
def predict_with_timestamps(request: CommentsWithTimestampRequest):

    comments_data = request.comments

    if not comments_data:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [item.text for item in comments_data]

        timestamps = [item.timestamp for item in comments_data]

        # Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # TF-IDF
        transformed_comments = (vectorizer.transform(preprocessed_comments))

        # Prediction
        predictions = model.predict(transformed_comments).tolist()

        predictions = [str(pred)for pred in predictions]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp
             in zip(comments,predictions,timestamps)]


# ==========================================
# Generate Pie Chart
# ==========================================

@app.post("/generate_chart")
def generate_chart(request: SentimentCountsRequest):

    try:
        sentiment_counts = (request.sentiment_counts)

        if not sentiment_counts:
            raise HTTPException(status_code=400, detail="No sentiment counts provided")

        labels = ["Positive","Neutral","Negative"]

        sizes = [int(sentiment_counts.get("1",0)),
                 int(sentiment_counts.get("0",0)),
                 int(sentiment_counts.get("-1",0))]

        if sum(sizes) == 0:
            raise HTTPException(status_code=400, detail="Sentiment counts sum to zero")

        colors = ["#36A2EB","#C9CBCF","#FF6384"]

        # Create chart
        plt.figure(figsize=(6, 6))

        plt.pie(sizes,labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, textprops={"color": "white"})

        plt.axis("equal")

        # Save in memory
        img_io = io.BytesIO()

        plt.savefig(img_io, format="PNG", transparent=True)

        img_io.seek(0)

        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error in /generate_chart: {e}")

        raise HTTPException(status_code=500,detail=("Chart generation failed: "f"{str(e)}"))


# ==========================================
# Generate Word Cloud
# ==========================================

@app.post("/generate_wordcloud")
def generate_wordcloud(request: CommentRequest):

    try:
        comments = request.comments

        if not comments:
            raise HTTPException(status_code=400, detail="No comments provided")

        # Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine comments
        text = " ".join(preprocessed_comments)

        if not text.strip():

            raise HTTPException(status_code=400, detail=("No valid text available " "after preprocessing"))

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")), collocations=False).generate(text)

        # Save image
        img_io = io.BytesIO()

        wordcloud.to_image().save(img_io, format="PNG")

        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")

    except HTTPException:
        raise

    except Exception as e:

        print(f"Error in /generate_wordcloud: {e}")

        raise HTTPException(status_code=500, detail=("Word cloud generation failed: "f"{str(e)}"))


# ==========================================
# Generate Sentiment Trend Graph
# ==========================================

@app.post("/generate_trend_graph")
def generate_trend_graph(request: SentimentDataRequest):

    try:

        sentiment_data = (request.sentiment_data)

        if not sentiment_data:

            raise HTTPException(status_code=400, detail="No sentiment data provided")

        # DataFrame
        df = pd.DataFrame(sentiment_data)

        # Timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Set timestamp index
        df.set_index("timestamp",inplace=True)

        # Numeric sentiment
        df["sentiment"] = (df["sentiment"].astype(int))

        # Sentiment labels
        sentiment_labels = {
            -1: "Negative",
            0: "Neutral",
            1: "Positive"}

        # Monthly counts
        monthly_counts = (df.resample("ME")["sentiment"].value_counts().unstack(fill_value=0))

        # Monthly totals
        monthly_totals = (monthly_counts.sum(axis=1))

        # Percentages
        monthly_percentages = (monthly_counts.T.div(monthly_totals).T * 100)

        # Ensure all columns exist
        for sentiment_value in [-1, 0, 1]:

            if (sentiment_value not in monthly_percentages.columns):
                monthly_percentages[sentiment_value] = 0

        # Sort columns
        monthly_percentages = (monthly_percentages[[-1, 0, 1]])

        # Create graph
        plt.figure(figsize=(12, 6))

        colors = {
            -1: "red",
            0: "gray",
            1: "green"}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(monthly_percentages.index, monthly_percentages[sentiment_value], 
                     marker="o", 
                     linestyle="-", 
                     label=sentiment_labels[ sentiment_value],
                     color=colors[sentiment_value])

        plt.title("Monthly Sentiment Percentage Over Time")

        plt.xlabel("Month")

        plt.ylabel("Percentage of Comments (%)")

        plt.grid(True)

        plt.xticks(rotation=45)

        # Date formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()

        plt.tight_layout()

        # Save graph
        img_io = io.BytesIO()

        plt.savefig(img_io, format="PNG")

        img_io.seek(0)

        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /generate_trend_graph: {e}")

        raise HTTPException(status_code=500, detail=("Trend graph generation failed: " f"{str(e)}"))


# ==========================================
# Run API
# ==========================================

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=5000)