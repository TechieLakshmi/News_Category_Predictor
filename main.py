from flask import Flask, render_template, request
import pandas as pd
from joblib import load
from app import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predictor',methods=["POST"])
def process():
    if request.method == 'POST':
        text = request.form['rawtext']
        text = [text]
        # load the saved pipleine model
        news_pipeline_tfid = load("pickle/news_tfid.joblib")
        news_pipeline_model = load("pickle/news_model.joblib")
        
        text_features = news_pipeline_tfid.transform(text)
        category = news_pipeline_model.predict(text_features)	
        for cat_id in category :
            y_pred_name = id_to_category[cat_id]
    return render_template("home.html",category = y_pred_name,text = text)

if __name__ == '__main__' :
    app.run(debug=True)
