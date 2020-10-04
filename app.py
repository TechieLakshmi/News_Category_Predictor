import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


#Reading the dataset
news_train = pd.read_csv('dataset/news_train.csv')

#Covert category into numerical index
news_train['category_id'] = news_train['Category'].factorize()[0]

#Creating a dictionary for category-id
category_id_news = news_train[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_news.values)
id_to_category = dict(category_id_news[['category_id', 'Category']].values)

#Stages of pipeline
pipeline_tfid = Pipeline(steps= [('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                                      min_df=5, 
                                                      norm='l2', 
                                                      encoding='latin-1', 
                                                      ngram_range=(1, 2), 
                                                      stop_words='english'))])
features = pipeline_tfid.fit_transform(news_train.Text).toarray()
labels = news_train.category_id
print(features.shape)

models = [
    RandomForestClassifier(n_estimators=200, max_depth=100, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy')
print("****************************")
print(accuracies)
print("****************************")
best_accuracies = max(accuracies)
pos = 0
for i in accuracies:
  pos += 1
  if (best_accuracies == i):
    break
best_model_position = pos -1
best_model = models[best_model_position]
print("****************************")
print(best_model )
print("****************************")

#Split Data 
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, news_train.index, test_size=0.33, random_state=42)

#Pipeline
pipeline_model = Pipeline(steps= [('model', best_model)])

#Train Algorithm
pipeline_model.fit(X_train, y_train)

# dump the pipeline model
dump(pipeline_tfid, filename="pickle/news_tfid.joblib")
dump(pipeline_model, filename="pickle/news_model.joblib")
