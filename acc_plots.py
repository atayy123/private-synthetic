import json
import matplotlib.pyplot as plt

with open('adult-results-k2.json') as json_file:
    data = json.load(json_file)

x = list(data["acc"].keys())
eps = [float(a) for a in x]
modes = [str(a) for a in range(1,6)]
markers = ["-v","-o","-.","-s","-h"]

for i,mode in enumerate(modes):
    y = []
    for ep in x:
        l = data["acc"][ep][mode]
        y.append(sum(l)/len(l))
    plt.plot(eps,y,markers[i])

plt.title("SVM Accuracy on Adult Dataset, $k=2$")
plt.xlabel("$\epsilon$")
plt.ylabel("Accuracy")
plt.legend(["PrivBayes","Greedy","Naive","PML-uniform","PML-empirical"])
plt.show()