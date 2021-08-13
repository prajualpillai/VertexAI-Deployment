import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.svm import SVC
import gcsfs
#from sklearn.externals import joblib
import joblib
from sklearn.preprocessing import label_binarize

df = pd.read_csv("proj.csv")

x = df.drop(["Room"],axis=1)

y = df["Room"]

x=df.iloc[:,:-1].values
y=df.iloc[:,7].values
y = label_binarize(y, classes=[1,2,3,4])

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)

clf_svm = SVC(kernel = 'rbf')
clf_svm.fit(X_train, y_train.argmax(axis=1))
BUCKET = "gs://test-1-bucket08/3"

#X_test.to_csv("test_set.csv")
#y_test.to_csv("test_ans.csv")

#pickle.dump(clf_svm,open('model.pkl','wb'))
joblib.dump(clf_svm,"model.joblib")
#print(list(X_test[10:11]))