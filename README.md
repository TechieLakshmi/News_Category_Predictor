# News Category Predictor

* Created a basic Flask based News Categorization Web App. Utilized Randomforest, MultinomialNB & Logistic regression for news categorization. 
* Used TF-IDF to transform text to vector.
* Using accuracy metric as performance metric
* In production level, we can use docker containerization for microservice architecture, scalability as we can replace the csv files with a database (MondoDb,Postgres etc..),casandra, SAP,Oracle,Salesforce etc.
* pip install -r requirements.txt before you run for library dependencies.
* Run the app using the code 'Python main.py'

## File Description

* news_train.csv - training set of 1490 records
* news_test.csv - test set of 736 records
* app.py - file for ML modeling & prediction
* main.py - flask endpoint creation
* home.html - page layout creation

## Data fields

* ArticleId - unique value given to the record
* Article - text of the header of article
* Category - category of the article (sports,tech,business,entertainment,politics)
