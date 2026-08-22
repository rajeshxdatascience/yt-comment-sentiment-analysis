FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY fast_api/ /app/

COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]