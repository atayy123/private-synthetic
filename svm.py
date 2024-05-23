from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import numpy as np

# Compare the classification accuracy of real and synthetic data
dir = 'yine/'
target = "income>50K"

original = pd.read_csv('data/adult.csv')
y = original[target]
X = original.drop(target,axis=1)

#scores = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# num = len(y)
# vals = y.value_counts()
# #print(vals[0]/num)
# #print(vals[1]/num)
# #pred = np.random.randint(2, size=len(y_test))
# pred = np.random.choice(2, len(y_test), p=[vals[0]/num,vals[1]/num])
# print(accuracy_score(y_test, pred))

clf = svm.SVC()
#clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = accuracy_score(y_test,pred)
#scores.append(score)
print('Model trained with real data: %.3f' % score)

dbs = os.listdir(dir)
for db in dbs:
    data = pd.read_csv(dir + db,index_col=0)
    y_temp = data[target]
    X_temp = data.drop(target,axis=1)

    X0, _, y0, _ = train_test_split(X_temp, y_temp, test_size=0.33, random_state=0)
    clf = svm.SVC()
    #clf = GradientBoostingClassifier()
    clf.fit(X0, y0)
    pred0 = clf.predict(X_test)
    score = accuracy_score(y_test,pred0)
    #scores.append(score)
    print(db +': %.3f' % score)

# labs = ['Real', 'other', 'No privacy', 'DP', 'Prior from data', 'Uniform']
# plt.bar(labs, scores)
# plt.title('PrivBayes tested on real data')
# plt.ylabel('Accuracy')
# plt.xlabel('Train dataset')
# plt.show()