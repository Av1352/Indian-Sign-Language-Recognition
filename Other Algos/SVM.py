import pickle
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sbn

data = pd.read_csv('csv files/train.csv', low_memory=False)
data.head()

test = pd.read_csv('csv files/test.csv', low_memory=False)
test.head()

x = data.iloc[:, 0:150]
y = data.iloc[:, -1]

le = LabelEncoder()
y = le.fit_transform(y)

x_test = test.iloc[:, 0:150]
y_test = test.iloc[:, -1]

y_test = le.fit_transform(y_test)

# Linear Kernel
cls = svm.SVC(kernel='linear')

cls.fit(x, y)

y_pred = cls.predict(x_test)
print("Accuracy score:")
print(metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(classification_report(y_test, y_pred))  # 150


with open('svm.pkl', 'wb') as files:
    pickle.dump(cls, files)

c_m = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(20, 17))
plt.title("Confusion Matrix for SVM")
df_cm = pd.DataFrame(c_m)
sbn.heatmap(df_cm, annot=True)
plt.savefig('svm.png')
