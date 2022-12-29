from typing import Type
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
import pandas as pd
import pickle as pkl


data = pd.read_csv("./data/emails.csv")

X = (data.iloc[:, 0:-1]).drop(["Email No."],axis=1)
y = data.iloc[:, -1]


train_x , test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=32)
clf_linear = svm.SVC(kernel="linear")
clf_linear.fit(train_x,train_y)

try:
  pkl.dump(clf_linear,open("model.pkl", "wb"))
except TypeError:
  print("didn't dump model")
  pass


def encode_text(text: str) -> pd.DataFrame:
    dataframe = pd.DataFrame(columns=train_x.columns)
    dataframe.columns = train_x.columns
    # initialize all null values
    dataframe.loc[0, :] =  np.zeros((1, len(dataframe.columns)), dtype=int)
    text_list = text.split()
    t= 0
    for i in text_list:
      if i in dataframe.columns:
        dataframe.at[0, i] = dataframe.at[0, i] + 1 
    return dataframe.copy()

def prediction(classifier , X, encode=False) -> int:
  if encode:
    t = encode_text(X)
  else:
    t = X.copy()
  return classifier.predict(t)

try:
  pkl.dump(encode_text,open("encode.pkl", "wb"))
  pkl.dump(prediction,open("pred.pkl", "wb"))
except TypeError:
  print("didn't dump encoder or prediction function")
  pass
