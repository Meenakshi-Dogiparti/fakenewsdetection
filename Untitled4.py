#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1️⃣ Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# 2️⃣ Load Dataset
from google.colab import files
uploaded = files.upload()

data = pd.read_csv('combined_news_data_cleaned.csv')  # Make sure you upload CLEANED dataset

# 3️⃣ Drop missing or duplicate rows (extra safety)
data = data.dropna(subset=['text', 'label']).drop_duplicates()

# 4️⃣ Encode Labels (Important Step!)
le = LabelEncoder()
data['label_encoded'] = le.fit_transform(data['label'])  # 0 and 1

# 5️⃣ Split Data
X = data['text'].values
y = data['label_encoded'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7️⃣ Model Training (With Balanced Class Weights to Avoid Biased Predictions)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vec, y_train)

# 8️⃣ Prediction & Evaluation
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9️⃣ Test Function for New News Text
def predict_news(news_text):
    news_vec = vectorizer.transform([news_text])
    prediction = model.predict(news_vec)
    label = le.inverse_transform(prediction)
    return label[0]

# 🔟 Example Prediction:
sample_news = "Government announces new policies for education sector."
print("Prediction for sample news:", predict_news(sample_news))


# In[ ]:


# 🔟 Example Prediction:
sample_news = "Government announces new policies for education sector."
print("Prediction for sample news:", predict_news(sample_news))

