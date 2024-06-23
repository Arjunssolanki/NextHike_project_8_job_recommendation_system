from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer# Download necessary NLTK data packages


# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('upwork.csv')
df = df[['title', 'is_hourly', 'hourly_low', 'hourly_high', 'budget', 'country']]
job_df = df.copy()

# Data Preprocessing
job_df['hourly_low'] = job_df['hourly_low'].fillna(job_df['hourly_low'].mean())
job_df['hourly_high'] = job_df['hourly_high'].fillna(job_df['hourly_high'].mean())
job_df['budget'] = job_df['budget'].fillna(job_df['budget'].mean())
job_df['title'] = job_df['title'].ffill()

# Normalize numerical features
scaler = MinMaxScaler()
job_df[['hourly_low', 'hourly_high', 'budget']] = scaler.fit_transform(job_df[['hourly_low', 'hourly_high', 'budget']])

# Text Cleaning Function
ps = PorterStemmer()
def cleaning(txt):
    txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
    tokens = nltk.word_tokenize(txt.lower())
    stemming = [ps.stem(w) for w in tokens if w not in stopwords.words('english')]
    return " ".join(stemming)

# Sample data and apply cleaning
job_df = job_df.sample(n=20000, random_state=42)
job_df['title'] = job_df['title'].astype(str).apply(lambda x: cleaning(x))

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['title'])
features = np.hstack((tfidf_matrix.toarray(), job_df[['hourly_low', 'hourly_high', 'budget']].values))

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(features, features)

# Recommendation function
def get_recommendations_by_title(title, cosine_sim=cosine_sim, top_n=10):
    title_tfidf = tfidf_vectorizer.transform([title])
    title_features = np.hstack((title_tfidf.toarray(), [[0, 0, 0]]))
    title_sim = cosine_similarity(title_features, features)
    sim_scores = list(enumerate(title_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    job_indices = [i[0] for i in sim_scores]
    return job_df.iloc[job_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    job_title = request.form.get('job_title')
    recommended_jobs = get_recommendations_by_title(job_title)
    return render_template('index.html', job_title=job_title, recommended_jobs=recommended_jobs.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
