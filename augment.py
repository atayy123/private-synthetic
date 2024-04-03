from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

# Try to increase the accuracy by using more data
original = pd.read_csv('data/adult.csv')
y = original['income>50K']
X = original.drop('income>50K',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

clf = svm.SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Model trained with real data: %.3f' % accuracy_score(y_test,pred))

data = pd.read_csv('synth/fake.csv', index_col=0)
y_temp = data['income>50K']
X_temp = data.drop('income>50K',axis=1)
X_aug = pd.concat([X_train,X_temp], axis=0)
y_aug = pd.concat([y_train,y_temp], axis=0)
clf = svm.SVC()
clf.fit(X_aug, y_aug)
pred = clf.predict(X_test)
print('Model trained with augmented data: %.3f' % accuracy_score(y_test,pred))