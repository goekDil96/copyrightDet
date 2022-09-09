import pickle 
import os
import json
import flask
from flask import Flask, render_template, request
import pandas as pd

from copyrightDet.match_string import MatchString
from sklearn.feature_extraction.text import TfidfVectorizer
from copyrightDet.rule_based import RuleBased


pkl_filename = "my_model.pkl"
pkl_vectorizer = "vectorizer.pkl"

rule_based = RuleBased()

# load models
with open(os.path.join(os.getcwd(), "data", pkl_vectorizer), "rb") as file:
    vectorizer = pickle.load(file)

with open(os.path.join(os.getcwd(), "data", pkl_filename), "rb") as file:
    pickle_model = pickle.load(file)

# init application
app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods = ['GET', 'POST'])
def index():
    data = None
    if request.method == 'POST':
        corpus = request.form["copy_data"].splitlines()
        # corpus = """ (C) 1996 dilara.goeksu@stud.h-da.de""".splitlines()

        result = rule_based.predict(corpus)
        X = vectorizer.fit_transform(corpus)
        df = pd.DataFrame(X.toarray(), index=corpus, columns=vectorizer.get_feature_names_out())
        y_pred_proba = pickle_model.predict_proba(df)[:, 1]
        y_pred = [int(result[i]) if result[i] != 0.5 else y_pred_proba[i] for i in range(len(result))]

        data = zip(corpus, y_pred)

    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run()