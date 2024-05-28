import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

with open('car-results-k2.json') as json_file:
    car2 = json.load(json_file)

with open('car-results-k3.json') as json_file:
    car3 = json.load(json_file)

with open('car-results-k4.json') as json_file:
    car4 = json.load(json_file)

with open('adult-results-2-k2.json') as json_file:
    adult2 = json.load(json_file)

fig, axs = plt.subplots(2, 2, figsize=(5.13,5.13))

x = list(car2["acc"].keys())
eps = [float(a) for a in x]
modes = [str(a) for a in range(1,6)]
markers = ["-v","-o","-.","-s","-h"]

for i,mode in enumerate(modes):
    y = []
    for ep in x:
        l = car2["acc"][ep][mode]
        y.append(sum(l)/len(l))
    axs[0,0].plot(eps,y,markers[i])
axs[0,0].set_title("Car, $k=2$")
axs[0,0].set_xlabel("$\epsilon$")
axs[0,0].set_ylabel("Accuracy")
#######
x = list(car3["acc"].keys())
eps = [float(a) for a in x]
modes = [str(a) for a in range(1,6)]
markers = ["-v","-o","-.","-s","-h"]

for i,mode in enumerate(modes):
    y = []
    for ep in x:
        l = car3["acc"][ep][mode]
        y.append(sum(l)/len(l))
    axs[0,1].plot(eps,y,markers[i])
axs[0,1].set_title("Car, $k=3$")
axs[0,1].set_xlabel("$\epsilon$")
#########
x = list(car4["acc"].keys())
eps = [float(a) for a in x]
modes = [str(a) for a in range(1,6)]
markers = ["-v","-o","-.","-s","-h"]

for i,mode in enumerate(modes):
    y = []
    for ep in x:
        l = car4["acc"][ep][mode]
        y.append(sum(l)/len(l))
    axs[1,0].plot(eps,y,markers[i])
axs[1,0].set_title("Car, $k=4$")
axs[1,0].set_xlabel("$\epsilon$")
axs[1,0].set_ylabel("Accuracy")
#########
x = list(adult2["acc"].keys())
eps = [float(a) for a in x]
modes = [str(a) for a in range(1,6)]
markers = ["-v","-o","-.","-s","-h"]

for i,mode in enumerate(modes):
    y = []
    for ep in x:
        l = adult2["acc"][ep][mode]
        y.append(sum(l)/len(l))
    axs[1,1].plot(eps,y,markers[i])
axs[1,1].set_title("Adult, $k=2$")
axs[1,1].set_xlabel("$\epsilon$")

fig.legend(["PrivBayes","Greedy","Naive","PML-uniform","PML-empirical"], ncol=3, loc = 'upper center', bbox_to_anchor = (0.05, 0.1, 1, 1))
plt.tight_layout()
plt.savefig("deneme.png", bbox_inches='tight')
# axs[0,0].title("Car, $k=2$")
# axs[0,0].xlabel("$\epsilon$")
# axs[0,0].ylabel("Accuracy")
# axs[0,0].legend(["PrivBayes","Greedy","Naive","PML-uniform","PML-empirical"])