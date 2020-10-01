import numpy as np
import pandas as pd
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
  # create 5 models with different 20% test sets, and store their accuracies
  accuracies = cross_val_score(model, features, labels, scoring='accuracy')
print("****************************")
print(accuracies)
print("****************************")

pipeline_model = Pipeline(steps= [('model', LogisticRegression(random_state=0))])

#Split Data 
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, news_train.index, test_size=0.33, random_state=42)

#Train Algorithm
pipeline_model.fit(X_train, y_train)

# Loading Test data
news_test = pd.read_csv('dataset/news_test.csv')
test_features = pipeline_tfid.transform(news_test.Text.tolist())

# Make Predictions
y_pred_proba = pipeline_model.predict_proba(test_features)
y_pred = pipeline_model.predict(test_features)
y_pred_name =[]
for cat_id in y_pred :
    y_pred_name.append(id_to_category[cat_id])
predicted_category = pd.DataFrame({
        "Category": y_pred_name,
        "News Text": news_test["Text"]
    })
print(predicted_category.head())

predicted_category.to_csv('dataset/news_predicted.csv', index=False)