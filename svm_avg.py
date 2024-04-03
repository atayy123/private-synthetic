from privbayes import privbayes_inference, privbayes_measurements
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import benchmarks
import random
import matplotlib.pyplot as plt
import numpy as np

original = pd.read_csv('data/adult.csv')
y = original['income>50K']
X = original.drop('income>50K',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

data, _ = benchmarks.adult_benchmark()
total = data.df.shape[0]

dp_scores = []
uniform_scores = []
real_scores = []

for ep in range(50):

    seed = random.randint(0,10000)
    measurements = privbayes_measurements(data, 1.0, seed, 'dp')
    est = privbayes_inference(data.domain, measurements, total=total)
    est = est.df
    y_temp = est['income>50K']
    X_temp = est.drop('income>50K',axis=1)
    X0, _, y0, _ = train_test_split(X_temp, y_temp, test_size=0.33)
    clf = svm.SVC()
    clf.fit(X0, y0)
    pred0 = clf.predict(X_test)
    dp_score = accuracy_score(y_test,pred0)
    dp_scores.append(dp_score)

    seed = random.randint(0,10000)
    m1 = privbayes_measurements(data, 1.0, seed, 'uniform')
    est1 = privbayes_inference(data.domain, m1, total=total)
    est1 = est1.df
    y1_temp = est1['income>50K']
    X1_temp = est1.drop('income>50K',axis=1)
    X1, _, y1, _ = train_test_split(X1_temp, y1_temp, test_size=0.33)
    clf = svm.SVC()
    clf.fit(X1, y1)
    pred1 = clf.predict(X_test)
    u_score = accuracy_score(y_test,pred1)
    uniform_scores.append(u_score)

    seed = random.randint(0,10000)
    m2 = privbayes_measurements(data, 1.0, seed, 'real')
    est2 = privbayes_inference(data.domain, m2, total=total)
    est2 = est2.df
    y2_temp = est2['income>50K']
    X2_temp = est2.drop('income>50K',axis=1)
    X2, _, y2, _ = train_test_split(X2_temp, y2_temp, test_size=0.33)
    clf = svm.SVC()
    clf.fit(X2, y2)
    pred2 = clf.predict(X_test)
    r_score = accuracy_score(y_test,pred2)
    real_scores.append(r_score)

    print('Run', ep+1)
    print('DP:', dp_score, 'Uniform:', u_score, 'Real:', r_score)


labs = ['DP', 'Uniform', 'Prior from data']

scores = [np.mean(dp_scores), np.mean(uniform_scores),np.mean(real_scores)]
plt.bar(labs, scores)
plt.title('PrivBayes')
plt.ylabel('Accuracy')
plt.xlabel('Train dataset')
plt.show()