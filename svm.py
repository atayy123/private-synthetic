from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

# Compare the classification accuracy of real and synthetic data

original = pd.read_csv('data/adult.csv')
y = original['income>50K']
X = original.drop('income>50K',axis=1)

scores = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

clf = svm.SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = accuracy_score(y_test,pred)
scores.append(score)
print('Model trained with real data: %.3f' % score)

dbs = os.listdir('./newest')
for db in dbs:
    if db == 'adult_syn_trial.csv':
        data = pd.read_csv('newest/' + db)
    else:
        data = pd.read_csv('newest/' + db,index_col=0)
    y_temp = data['income>50K']
    X_temp = data.drop('income>50K',axis=1)

    X0, _, y0, _ = train_test_split(X_temp, y_temp, test_size=0.33, random_state=0)
    clf = svm.SVC()
    clf.fit(X0, y0)
    pred = clf.predict(X_test)
    score = accuracy_score(y_test,pred)
    scores.append(score)
    print(db +': %.3f' % score)

# labs = ['Real', 'other', 'No privacy', 'DP', 'Prior from data', 'Uniform']
# plt.bar(labs, scores)
# plt.title('PrivBayes tested on real data')
# plt.ylabel('Accuracy')
# plt.xlabel('Train dataset')
# plt.show()