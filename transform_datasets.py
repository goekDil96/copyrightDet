#
# SPDX-FileCopyrightText: Copyright 2022 Dilara Göksu
#

import os 
import json
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.getcwd(), "copyrightDet"))
from copyrightDet.match_string import MatchString

for path in ["pos_neg_copy_y_train", "pos_neg_copy_y_test1", "pos_neg_copy_y_test2"]:
        with open(os.path.join(os.getcwd(), "data", f"{path}.json"), "r", encoding="utf8") as file:
                data = json.load(file)
        
        corpus = data.keys()

        with open(os.path.join(os.getcwd(), "data", "vocabulary.txt"), "r", encoding="utf8") as file:
                vocabulary = file.read().splitlines()

        vectorizer = TfidfVectorizer(preprocessor=MatchString().preprocess,
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
        df["score"] = data.values()
        # df.drop_duplicates(inplace=True)
        df.to_csv(os.path.join(os.getcwd(), "data", f"{path}.csv"), encoding="utf8")