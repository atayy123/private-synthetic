from ektelo.privBayes import privBayesSelect
from mbi import Dataset
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import json
import itertools

# This code calculates the SVM accuracy of PrivBayes, PrivBayes-PML, GreedyBayes, NaiveBayes
# with the given range of epsilons for several runs (epochs). The results are written in a
# JSON file.

# range of epsilons
eps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 4.65]
# number of iterations
epochs = 10

#pgm_iters = 10000

# max number of parents
k = 3

# info about the dataset
dataset = "car"
target = "class"
dataset_name = "data/" + dataset + ".csv"
dataset_domain = "data/" + dataset + "-domain.json"


with open(dataset_name) as f:
    first_line = f.readline().rstrip()
colnames = first_line.split(",")

# form test dataset from original data
original = pd.read_csv(dataset_name)
train, test = train_test_split(original, test_size=0.33)
y_test = test[target]
X_test = test.drop(target,axis=1)
train.to_csv("temp.csv")

data = Dataset.load("temp.csv", dataset_domain)
total = data.df.shape[0]
domain = data.domain
config = ''
for a in domain:
    values = [str(i) for i in range(domain[a])]
    config += 'D ' + ' '.join(values) + ' \n'
config = config.encode('utf-8')

values = np.ascontiguousarray(data.df.values.astype(np.int32))

# create the results dict
all_results = {"acc":{}, "tvd":{}}

# iterate epsilon
for ep in eps:
    results = {}
    tvd_results = {}
    # iterate epochs
    for i in range(epochs):
        print("Eps:", ep, "Epoch:", i+1, "started")
        seed = np.random.randint(0,10000)
        # different modes
        for mode in range(1,6):
            # start PrivBayes algorithm
            ans = privBayesSelect.py_get_model(values, config, ep, k+1, seed, mode)
            # read the produced csv file
            est = pd.read_csv("Ep"+f'{ep:.6f}'+"Mode"+str(mode)+".csv", index_col=0, names=colnames)
            # run SVM classification
            y_temp = est[target]
            X_temp = est.drop(target,axis=1)
            #X0, _, y0, _ = train_test_split(X_temp, y_temp, test_size=0.33)
            clf = svm.SVC()
            clf.fit(X_temp, y_temp)
            # predict on original data
            pred0 = clf.predict(X_test)
            # get score
            score = accuracy_score(y_test,pred0)
            # add to dict
            if mode in results:
                results[mode].append(score)
            else:
                results[mode] = [score]

            # average TVD of all 3-way marginals
            tvds = []
            # get all 3-way marginals
            for marg in itertools.combinations(colnames, 3):
                marg = list(marg)
                org_dist = original.value_counts(marg,normalize=True)
                synth_dist = est.value_counts(marg,normalize=True)
                # dist difference
                diff_dist = org_dist-synth_dist
                diff_dist.dropna(inplace=True)
                tvd = 0.5*np.sum(np.abs(diff_dist))
                #print(tvd)
                tvds.append(tvd)

            if mode in tvd_results:
                tvd_results[mode].append(np.mean(tvds))
            else:
                tvd_results[mode] = [np.mean(tvds)]
    
    # save the results
    all_results["acc"][ep] = results
    all_results["tvd"][ep] = tvd_results

# write results to json
with open(dataset + "-results-Rk" + str(k) +".json", "w") as outfile: 
    json.dump(all_results, outfile, indent=5)

# ans = ans.decode('utf-8')[:-1]
# ans = ans.split("\n")

# projections = []
# measurements = []
