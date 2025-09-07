📰 News Article Classification

Natural Language Processing (NLP) Project

📖 Overview

In today’s digital world, vast amounts of news articles are published daily across platforms. Efficient classification into categories such as sports, politics, and technology is crucial for:

News organizations 📰

Social media platforms 📱

Aggregators 🔎

This project develops a machine learning model that classifies news articles into predefined categories, enabling better content management, personalization, and recommendations.

❓ Problem Statement

The objective is to build a robust classifier that automatically categorizes news articles into sports, politics, technology (and other categories).

Goals:

✔️ Collect and preprocess labeled news articles.

✔️ Extract meaningful features (TF-IDF, Word2Vec, BoW).

✔️ Train ML classifiers (Logistic Regression, Naive Bayes, SVM).

📂 Dataset Information

Dataset Name: data_news.csv

Columns:

headline → News headline

short_description → Short description of the article

category → Labeled category (sports, politics, technology, etc.)

Processing:

Combined headline + short_description into one column → text

⚙️ Methodology
🔹 1. Data Preprocessing

Removed duplicates & missing values

Combined headline + short_description

Converted text to lowercase

Removed punctuation, numbers, and stopwords

Applied lemmatization using NLTK

🔹 2. Feature Extraction

TF-IDF Vectorization

Bag-of-Words representation

Word2Vec embeddings

Exploratory Data Analysis (EDA) → checked distribution of article categories

🔹 3. Model Development

Trained multiple classification models:

Logistic Regression

Naive Bayes

Support Vector Machines (SVM)

Applied cross-validation and hyperparameter tuning for optimization.

🔹 4. Model Evaluation

Metrics: Accuracy, Precision, Recall, F1-score

Visualizations: Confusion Matrices, Bar Charts

Compared model performances to select the best classifier

✔️ Evaluate model performance and provide insights.

📊 Results & Insights

Classifiers achieved strong performance in distinguishing between sports, politics, and technology articles.

TF-IDF with Logistic Regression/SVM gave the best accuracy.

Most frequent and discriminative keywords per category were identified.

Automated classification reduces manual tagging efforts and improves personalization.

🛠️ Tools & Libraries

Python 

Data Handling: pandas, numpy

NLP Preprocessing: NLTK, spaCy

Feature Extraction: scikit-learn (TF-IDF, BoW), gensim (Word2Vec)

Models: Logistic Regression, Naive Bayes, SVM

Visualization: matplotlib, seaborn

Environment: Jupyter Notebook





