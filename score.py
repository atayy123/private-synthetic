import json
import matplotlib.pyplot as plt

# compare the score functions for Car dataset
k_s = [2,3,4]
metrics = ["acc", "tvd"]

fig, axs = plt.subplots(2, 3, figsize=(5.5,5), sharex="col", sharey="row")

for i,k in enumerate(k_s):
    with open('car-results-k'+str(k)+'.json') as json_file:
        data_normal = json.load(json_file)

    with open('car-results-R-k'+str(k)+'.json') as json_file2:
        data_r = json.load(json_file2)

    x = list(data_normal["acc"].keys())
    eps = [float(a) for a in x]

    mi_score = []
    for ep in x:
        l = data_normal["acc"][ep]["4"]
        mi_score.append(sum(l)/len(l))
    axs[0,i].plot(eps,mi_score,"-*")

    r_score  = []
    for ep in x:
        l = data_r["acc"][ep]["4"]
        r_score.append(sum(l)/len(l))
    axs[0,i].plot(eps,r_score,"-o")
########################################
    mi_score = []
    for ep in x:
        l = data_normal["tvd"][ep]["4"]
        mi_score.append(sum(l)/len(l))
    axs[1,i].plot(eps,mi_score,"-*")

    r_score  = []
    for ep in x:
        l = data_r["tvd"][ep]["4"]
        r_score.append(sum(l)/len(l))
    axs[1,i].plot(eps,r_score,"-o")
axs[0,0].set_ylabel("Accuracy")
axs[1,0].set_ylabel("TVD")
axs[0,0].set_title("$k=2$")
axs[0,1].set_title("$k=3$")
axs[0,2].set_title("$k=4$")
axs[1,0].set_xlabel("$\epsilon$")
axs[1,1].set_xlabel("$\epsilon$")
axs[1,2].set_xlabel("$\epsilon$")

plt.tight_layout()
fig.legend(["MI", "R"], ncol=2, loc = 'upper center', bbox_to_anchor = (0.05, 0.05, 1, 1))
plt.savefig("mi_or_r.png", bbox_inches='tight')