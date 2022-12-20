import pandas as pd

data = pd.read_csv("./data/emails.csv")

X = (data.iloc[:, 0:-1]).drop(["Email No."],axis=1)
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split

train_x , test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=32)

sample = test_y[2020]

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from sklearn import svm

clf_linear = svm.SVC(kernel="linear")

clf_linear.fit(train_x,train_y)
preds_linear = clf_linear.predict(test_x)
report_linear = classification_report(test_y, preds_linear)

print(report_linear)
linear_mat =confusion_matrix(test_y,preds_linear)

clf_poly = svm.SVC(kernel="poly")

clf_poly.fit(train_x,train_y)
preds_poly = clf_poly.predict(test_x)
report_poly = classification_report(test_y, preds_poly)

print(report_poly)
ploy_mat =confusion_matrix(test_y,preds_poly)

clf_rbf = svm.SVC(kernel="rbf")

clf_rbf.fit(train_x,train_y)
preds_rbf = clf_rbf.predict(test_x)
report_rbf = classification_report(test_y, preds_rbf)


print(report_rbf)
rbf_mat =confusion_matrix(test_y,preds_rbf)

from sklearn import metrics

metrics.ConfusionMatrixDisplay(rbf_mat, display_labels="rbf").plot()
metrics.ConfusionMatrixDisplay(ploy_mat, display_labels="ploy").plot()
metrics.ConfusionMatrixDisplay(linear_mat, display_labels="linaer").plot()

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
