#
# SPDX-FileCopyrightText: Copyright 2022 Dilara Göksu
#

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "copyrightDet"))
import json
import pandas as pd

from copyrightDet.match_string import MatchString
from copyrightDet.rule_based import RuleBased

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from pypmml import Model


def main():
    prePro = MatchString()

    with open(os.path.join(os.getcwd(), "data", "pos_neg_copy_y_test2.json"), "r", encoding="utf8") as file:
        data = json.load(file)

    corpus = data.keys()

    rule_based = RuleBased()
    result = rule_based.predict(corpus)

    with open(os.path.join(os.getcwd(), "data", "vocabulary.txt"), "r", encoding="utf8") as file:
        vocabulary = file.read().splitlines()

    vectorizer = TfidfVectorizer(preprocessor=prePro.preprocess,
                                 vocabulary=vocabulary,
                                 ngram_range=(1, 4)
                                 )

    # X = vectorizer.fit_transform(corpus)

    X_test = vectorizer.fit_transform(data.keys())

    print("Feature names:")
    print(vectorizer.get_feature_names_out())

    print("\n")
    print("DataFrame:")
    df = pd.DataFrame(X_test.toarray(), index=corpus, columns=vectorizer.get_feature_names_out())

    clf = Model.load(os.path.join(os.getcwd(), "data", "gradboos_knime.pmml"))
    # clf = GradientBoostingClassifier(random_state=0, max_depth=4, learning_rate=0.05, n_estimators=190).fit(df, list(data.values()))
    y_pred_proba = clf.predict(df).values[:,1]
    y_pred_or = [0 if y_pred_proba[i] < 0.3839676779502669 else 1 for i in range(len(y_pred_proba))]

    y_pred = [int(result[i]) if result[i] != 0.5 else y_pred_or[i] for i in range(len(result))]

    ConfusionMatrixDisplay.from_predictions(list(data.values()), y_pred)

    plt.show()

if __name__ == "__main__":
    main()
