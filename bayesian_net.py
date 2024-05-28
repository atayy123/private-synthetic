from ektelo.privBayes import privBayesSelect
from mbi import Dataset, FactoredInference
import numpy as np
from ektelo.matrix import Identity
from privbayes import privbayes_inference
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.metrics import log_likelihood_score
from pgmpy.models import BayesianNetwork

seed = 0
eps = 0.5
iters = 10000
# max number of parents
k = 2
dataset_name = "data/car.csv"
dataset_domain = "data/car-domain.json"

data = Dataset.load(dataset_name, dataset_domain)
total = data.df.shape[0]
domain = data.domain
config = ''
for a in domain:
    values = [str(i) for i in range(domain[a])]
    config += 'D ' + ' '.join(values) + ' \n'
config = config.encode('utf-8')

values = np.ascontiguousarray(data.df.values.astype(np.int32))

# start PrivBayes algorithm

# DP
#ans = privBayesSelect.py_get_model(values, config, eps, k+1, seed, 1)
# no privacy
ans = privBayesSelect.py_get_model(values, config, eps, k+1, seed, 2)
# naive
#ans = privBayesSelect.py_get_model(values, config, eps, k+1, seed, 3)
# PML, uniform prior
#ans = privBayesSelect.py_get_model(values, config, eps, k+1, seed, 4)
# PML, empirical prior
#ans = privBayesSelect.py_get_model(values, config, eps, k+1, seed, 5)


ans = ans.decode('utf-8')[:-1]
ans = ans.split("\n")

projections = []
measurements = []

# initialize directed graph
G = nx.DiGraph()

for i in range(0, len(ans), 2):
    m = ans[i]
    p = [domain.attrs[int(a)] for a in m.split(',')[::2]]
    #print(p)
    projections.append(tuple(p))

    if len(p) == 1:
        G.add_node(p[0])
    else:
        node = p.pop(0)
        xx = [(aa, node) for aa in p]
        G.add_edges_from(xx)
    
    counts_str = ans[i+1].split(" ")[:-1]
    counts = [float(c) for c in counts_str]

    #I = Identity(len(counts))
   # measurements.append( (I, counts, 1.0, p) )
    #print(p)
    #print(counts)

#print("Log-likelihood:", log_likelihood_score(G, data.df))
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, font_color="black")
plt.show()

# # elim_order = [m[3][0] for m in measurements][::-1]
# # engine = FactoredInference(data.domain, iters=iters, warm_start=False, elim_order=elim_order)
# # est3 = engine.estimate(measurements, total=total)
# # synth3 = est3.synthetic_data(rows=total)
# # synth3.df.to_csv('privbayes_pgm.csv')



# # # generate a dataset from the Bayesian model and histogram measurements
# # est = privbayes_inference(data.domain, measurements, total=total)
# # est.df.to_csv("car_priv.csv")