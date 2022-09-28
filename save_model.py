#
# SPDX-FileCopyrightText: Copyright 2022 Dilara Göksu
#

import os
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import nltk
import json
from copyrightDet.match_string import MatchString
from copyrightDet.rule_based import RuleBased
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    prePro = MatchString()

    with open(os.path.join(os.getcwd(), "data", "pos_neg_copy_y_train.json"), "r", encoding="utf8") as file:
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
    scaler = MaxAbsScaler()
    scaler.fit(X_test)

    print("Feature names:")
    print(vectorizer.get_feature_names_out())

    print("\n")
    print("DataFrame:")
    df = pd.DataFrame(X_test.toarray(), index=corpus, columns=vectorizer.get_feature_names_out())
    # df["score"] = data.values()
    # df.drop_duplicates(inplace=True)
    # df.to_csv(os.path.join(os.getcwd(), "data", "pos_neg_copy_y__all1_processed.csv"))

    # clf = LogisticRegression(random_state=0, C=100, max_iter=100000000).fit(df, list(data.values()))
    clf = GradientBoostingClassifier(random_state=0, learning_rate=0.91, min_samples_leaf=6).fit(df, list(data.values()))
    print(vars(clf))

    pkl_filename = "vectorizer.pkl"
    with open(os.path.join(os.getcwd(), "data", pkl_filename), "wb") as file:
        pickle.dump(vectorizer, file)

    pkl_filename = "my_model.pkl"
    with open(os.path.join(os.getcwd(), "data", pkl_filename), "wb") as file:
        pickle.dump(clf, file)


if __name__ == "__main__":
    main()
