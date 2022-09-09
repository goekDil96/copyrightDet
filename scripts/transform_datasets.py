import os 
import json
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.getcwd(), "copyrightDet"))
from match_string import MatchString

for path in ["pos_neg_copy_y_train", "pos_neg_copy_y_test1", "pos_neg_copy_y_test2"]:
        with open(os.path.join(os.getcwd(), "data", f"{path}.json"), "r", encoding="utf8") as file:
                data = json.load(file)
        
        corpus = data.keys()

        vocabulary = ["copyright",
                  "corp", 
                  "corporation", 
                  "ltd", 
                  "inc",
                  "authors", 
                  "author",  
                  "contributor", 
                  "all rights reserved", 
                  "group", 
                  "company", 
                  "institute", 
                  "contributors", 
                  "developer", 
                  "developers", 
                  "foundation", 
                  "affiliates", 
                  "limited", 
                  "se", 
                  "others", 
                  "enterprise", 
                  "incorporated", 
                  "co", 
                  "llc", 
                  "detected_url", 
                  "detected_email", 
                  "detected_org", 
                  "detected_person", 
                  "detected_year",
                  "detected_copyright detected_year",
                  "detected_copyright detected_year detected_year",
                  "detected_copyright detected_copyright detected_year",
                  "detected_copyright detected_copyright detected_year detected_year",
                  "gmbh",
                  "by",
                  "detected_copyright detected_copyright detected_org detected_year",
                  "detected_copyright detected_org detected_year",
                  "detected_copyright detected_copyright detected_person detected_year",
                  "detected_copyright detected_person detected_year",
                  "the",
                  "and or its affiliates",
                  "original author or authors",
                  "detected_other_words",
                  "detected_copyright detected_org",
                  "detected_copyright detected_copyright detected_org"
                  "detected_copyright detected_person",
                  "detected_copyright detected_copyright detected_person",
                  "word_between_copyright",
                  ]

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
        df.drop_duplicates(inplace=True)
        df.to_csv(os.path.join(os.getcwd(), "data", f"{path}.csv"), encoding="utf8")