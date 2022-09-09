from sklearn.model_selection import train_test_split
import os 
import json
import pandas as pd


with open(os.path.join(os.getcwd(), "data", "pos_neg_copy_y.json"), "r", encoding="utf8") as file:
        data = json.load(file)

X_train, X_test, y_train, y_test = train_test_split(list(data.keys()), list(data.values()), stratify=list(data.values()), test_size=0.6)

# save train data
df = pd.DataFrame(y_train, index=X_train, columns=["score"])
df.T.to_json(os.path.join(os.getcwd(), "data", "pos_neg_copy_y_train.json"), orient="records")

del X_train
del y_train

X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5)

# save train data
df = pd.DataFrame(y_test1, index=X_test1, columns=["score"])
df.T.to_json(os.path.join(os.getcwd(), "data", "pos_neg_copy_y_test1.json"), orient="records")

# save train data
df = pd.DataFrame(y_test2, index=X_test2, columns=["score"])
df.T.to_json(os.path.join(os.getcwd(), "data", "pos_neg_copy_y_test2.json"), orient="records")
