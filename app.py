import pickle 
import os
import json
import flask
from flask import Flask, render_template, request
import pandas as pd

from copyrightDet.match_string import MatchString
from sklearn.feature_extraction.text import TfidfVectorizer
from copyrightDet.rule_based import RuleBased

from pypmml import Model


pkl_filename = "my_model.pkl"
pkl_vectorizer = "vectorizer.pkl"

rule_based = RuleBased()

# load models
with open(os.path.join(os.getcwd(), "data", pkl_vectorizer), "rb") as file:
    vectorizer = pickle.load(file)

# with open(os.path.join(os.getcwd(), "data", pkl_filename), "rb") as file:
#     pickle_model = pickle.load(file)

model = Model.load(os.path.join(os.getcwd(), "data", "gradboos_knime.pmml"))

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
        y_pred_proba = model.predict(df).values[:, 1]
        y_pred_or = [0 if y_pred_proba[i] < 0.3839676779502669 else 1 for i in range(len(y_pred_proba))]
        y_pred = [int(result[i]) if result[i] != 0.5 else y_pred_or[i] for i in range(len(result))]

        data = zip(corpus, y_pred)

    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run()