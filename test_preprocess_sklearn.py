import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "copyrightDet"))
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import json
from copyrightDet.match_string import MatchString
from copyrightDet.rule_based import RuleBased
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn_pmml_model.ensemble import PMMLGradientBoostingClassifier
from pypmml import Model


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

    print("Feature names:")
    print(vectorizer.get_feature_names_out())

    print("\n")
    print("DataFrame:")
    df = pd.DataFrame(X_test.toarray(), index=corpus, columns=vectorizer.get_feature_names_out())
    # df["score"] = data.values()
    # df.drop_duplicates(inplace=True)
    # df.to_csv(os.path.join(os.getcwd(), "data", "pos_neg_copy_y__all1_processed.csv"))
    
    # clf = PMMLGradientBoostingClassifier("data/gradboos_knime.pmml")
    clf = Model.load('data/gradboos_knime.pmml')
    # clf = GradientBoostingClassifier(random_state=0, max_depth=4, learning_rate=0.05, n_estimators=190).fit(df, list(data.values()))
    y_pred_proba = clf.predict_proba(df)
    y_pred_or = clf.predict(df)

    y_pred = [int(result[i]) if result[i] != 0.5 else y_pred_or[i] for i in range(len(result))]
    print(clf.score(df, list(data.values())))

    ConfusionMatrixDisplay.from_estimator(clf, df, list(data.values()))
    # ConfusionMatrixDisplay.from_predictions(list(data.values()), y_pred)

    for i in range(len(list(data.keys()))):
        if list(data.values())[i] == 0 and y_pred[i] >= 0.5:
            print(list(data.keys())[i], list(data.values())[i], y_pred[i])
        if list(data.values())[i] == 1 and y_pred[i] <= 0.5:
            print(list(data.keys())[i], list(data.values())[i], y_pred[i])


    plt.show()

    plt.scatter(y_pred_proba[:,1], list(data.values()))
    plt.show()

    for i in [1, 2]:
        with open(os.path.join(os.getcwd(), "data", f"pos_neg_copy_y_test{i}.json"), "r", encoding="utf8") as file:
            data1 = json.load(file)

        result = rule_based.predict(data1.keys())
        
        X_train = vectorizer.fit_transform(data1.keys())

        df1 = pd.DataFrame(X_train.toarray(), index=data1.keys(), columns=vectorizer.get_feature_names_out())
        y_pred_or = clf.predict(df1)

        y_pred = [int(result[i]) if result[i] != 0.5 else y_pred_or[i] for i in range(len(result))]

        y_pred_proba = clf.predict_proba(df1)
        print(clf.score(df1, list(data1.values())))
        # ConfusionMatrixDisplay.from_estimator(clf, df1, list(data1.values()))
        ConfusionMatrixDisplay.from_predictions(list(data1.values()), y_pred)

        for i in range(len(list(data1.keys()))):
            if list(data1.values())[i] == 0 and y_pred[i] >= 0.5:
                print(list(data1.keys())[i], list(data1.values())[i], y_pred[i])
            if list(data1.values())[i] == 1 and y_pred[i] <= 0.5:
                print(list(data1.keys())[i], list(data1.values())[i], y_pred[i])


        plt.show()

        plt.scatter(y_pred_proba[:,1], list(data1.values()))
        plt.show()


if __name__ == "__main__":
    main()
