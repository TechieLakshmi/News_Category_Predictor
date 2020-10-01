from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    df = pd.read_csv('dataset/news_predicted.csv')
    return render_template('home.html', tables=[df.to_html(classes='data',index = False)], titles=df.columns.values)

if __name__ == '__main__' :
    app.run(debug=True)