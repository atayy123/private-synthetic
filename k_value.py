# which k value work better for Car dataset
import json
import matplotlib.pyplot as plt

metric = "acc"
markers = ["-*", "-o", "-^"]

for k in range(2,5):
    with open('car-results-k'+str(k)+'.json') as json_file:
        data = json.load(json_file)
    x = list(data[metric].keys())
    eps = [float(a) for a in x]
    score = []
    for ep in x:
        l = data[metric][ep]["4"]
        score.append(sum(l)/len(l))
    plt.plot(eps,score,markers.pop())
plt.legend(["2","3","4"])
plt.show()