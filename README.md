# 🎬 Rotten Tomatoes Movie Reviews -- Sentiment Analysis (NLP Project)

## 📌 Overview

This project performs **sentiment analysis** on Rotten Tomatoes movie
reviews using **Natural Language Processing (NLP)** and **Machine
Learning**.\
The goal is to classify reviews as **Positive** or **Negative**.

-   Environment: **Google Colab**\
-   Input: CSV file with at least two columns:
    -   `review` → movie review text\
    -   `label` → sentiment (`positive/negative` or `1/0`)\
-   Output: Trained ML model that predicts review sentiment.

------------------------------------------------------------------------

## ⚙️ Project Steps

### 1. Setup Environment

``` bash
!pip install -q nltk scikit-learn pandas matplotlib joblib
```

Import libraries (NLTK, sklearn, pandas, matplotlib, joblib).\
Download NLTK resources (`punkt`, `stopwords`, `wordnet`).

------------------------------------------------------------------------

### 2. Load Dataset from Google Drive

``` python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/data/rotten_tomatoes_reviews.csv')
```

------------------------------------------------------------------------

### 3. Normalize Labels

-   Ensure dataset has `review` and `label` columns.\
-   Convert labels to numeric:
    -   Positive → `1`\
    -   Negative → `0`

------------------------------------------------------------------------

### 4. Text Preprocessing

-   Convert text to lowercase\
-   Remove punctuation, numbers, HTML tags, URLs\
-   Tokenize\
-   Remove stopwords\
-   Lemmatize words

Cleaned reviews are stored in `clean_review` column.

------------------------------------------------------------------------

### 5. Train/Test Split

-   80% training, 20% testing\
-   Stratified split for balanced labels

------------------------------------------------------------------------

### 6. Feature Extraction

Use **TF-IDF Vectorizer** with: - `max_features=8000`\
- `ngram_range=(1,2)`

------------------------------------------------------------------------

### 7. Model Training

Trained 3 classifiers: - Logistic Regression\
- Naive Bayes\
- Linear SVC

Best model is selected by accuracy.

------------------------------------------------------------------------

### 8. Evaluation

-   Accuracy, Precision, Recall, F1-score\
-   Confusion Matrix (console + heatmap)

------------------------------------------------------------------------

### 9. Save Model

Save the best model and vectorizer to Google Drive:

``` python
joblib.dump(best_model, '/content/drive/MyDrive/data/best_sentiment_model.joblib')
joblib.dump(vectorizer, '/content/drive/MyDrive/data/tfidf_vectorizer.joblib')
```

------------------------------------------------------------------------

### 10. Prediction (Example)

``` python
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = best_model.predict(vec)[0]
    return 'positive' if pred==1 else 'negative'

print(predict_sentiment("This movie was absolutely fantastic!"))
print(predict_sentiment("A boring and terrible experience."))
```

------------------------------------------------------------------------

## 📊 Results

-   Expected accuracy: **80--90%** (depends on dataset quality &
    preprocessing).\
-   Example:
    -   `"An outstanding movie with superb acting!"` → **Positive**\
    -   `"The film was dull and boring."` → **Negative**

------------------------------------------------------------------------

## 🚀 Next Steps (Optional)

-   Tune hyperparameters with GridSearchCV\
-   Try **Word2Vec** / **BERT** for better embeddings\
-   Expand to multi-class sentiment (e.g., Very Positive, Neutral, Very
    Negative)

------------------------------------------------------------------------

## 📄 GitHub Repository Description

Lightweight NLP project for **sentiment analysis of Rotten Tomatoes
movie reviews**.\
Implements text preprocessing, TF-IDF feature extraction, and ML
classifiers (Logistic Regression, Naive Bayes, Linear SVC).\
Runs fully in **Google Colab** and saves trained models to Google Drive.

------------------------------------------------------------------------

👨‍💻 Author: Mohammad Ashraful Alam
