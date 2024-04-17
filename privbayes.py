from ektelo.privBayes import privBayesSelect
import numpy as np
from mbi import Dataset, Factor, FactoredInference, mechanism
# from mbi.dataset import Dataset
# from mbi.factor import Factor
# from mbi.inference import FactoredInference
from ektelo.matrix import Identity
import pandas as pd
import itertools
import argparse
import benchmarks

"""
This file implements PrivBayes, with and without our graphical-model based inference.

Zhang, Jun, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. "Privbayes: Private data release via bayesian networks." ACM Transactions on Database Systems (TODS) 42, no. 4 (2017): 25.
"""


def privbayes_measurements(data, eps=1.0, seed=0, mode='dp'):
    domain = data.domain
    config = ''
    for a in domain:
        values = [str(i) for i in range(domain[a])]
        config += 'D ' + ' '.join(values) + ' \n'
    config = config.encode('utf-8')
    
    values = np.ascontiguousarray(data.df.values.astype(np.int32))
    
    if mode == 'off':
        ans = privBayesSelect.py_get_model(values, config, eps, 1.0, seed, 2)
    elif mode == 'dp':
        ans = privBayesSelect.py_get_model(values, config, eps, 1.0, seed, 1)
    elif mode == 'uniform':
        ans = privBayesSelect.py_get_model(values, config, eps, 1.0, seed, 4)
    elif mode == 'real':
        ans = privBayesSelect.py_get_model(values, config, eps, 1.0, seed, 5)

    ans = ans.decode('utf-8')[:-1]
    
    projections = []
    for m in ans.split('\n'):
        p = [domain.attrs[int(a)] for a in m.split(',')[::2]]
        projections.append(tuple(p))
        
    prng = np.random.RandomState(seed) 
    measurements = []
    delta = len(projections)
    for proj in projections:
        x = data.project(proj).datavector()
        print(proj,len(x))
       # print(x)
        I = Identity(x.size)
        budget = eps/(2*delta)
        if mode == 'dp': # differential privacy
            noise = prng.laplace(loc=0, scale=2/budget, size=x.size)
            # if len(proj) == 2:
            #     print(mode,proj)
            #     print(type(proj))
            #     print(2/budget)
            #     print(noise)
        elif mode == 'off':
            noise = np.zeros(x.size)
        else:
            if mode == 'uniform': # PML with uniform prior
                p_min = 1/len(x)
            elif mode == 'real': # PML with priors taken from real data
                p_min = np.min(x[np.nonzero(x)])/np.sum(x)

            if -np.log(p_min) >= budget:
                b = 2/(budget+np.log((1-p_min)/(1-p_min*np.exp(budget))))
                noise = prng.laplace(loc=0, scale=b, size=x.size)
                # if len(proj) == 2:
                #     print(mode, proj)
                #     print(b)
                #     print(p_min)
                #     print(noise)
            else:
                print('no noise')
                noise = np.zeros(x.size)

        y = x + noise
        #print(I.dot(x))
        measurements.append( (I, y, 1.0, proj) )
     
    return measurements

def privbayes_inference(domain, measurements, total):
    synthetic = pd.DataFrame()

    _, y, _, proj = measurements[0]
    y = np.maximum(y, 0)
    y /= y.sum()
    col = proj[0]
    synthetic[col] = np.random.choice(domain[col], total, True, y)
        
    for _, y, _, proj in measurements[1:]:
        # find the CPT
        col, dep = proj[0], proj[1:]
       # print(col)
        y = np.maximum(y, 0)
        dom = domain.project(proj)
        cpt = Factor(dom, y.reshape(dom.shape))
        marg = cpt.project(dep)
        cpt /= marg
        cpt2 = np.moveaxis(cpt.project(proj).values, 0, -1)
        
        # sample current column
        synthetic[col] = 0
        rng = itertools.product(*[range(domain[a]) for a in dep])
        for v in rng:
            idx = (synthetic.loc[:,dep].values == np.array(v)).all(axis=1)
            p = cpt2[v].flatten()
            if p.sum() == 0:
                p = np.ones(p.size) / p.size
            n = domain[col]
            N = idx.sum()
            if N > 0:
                synthetic.loc[idx,col] = np.random.choice(n, N, True, p)

    return Dataset(synthetic, domain)

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = 'adult'
    params['iters'] = 10000
    params['epsilon'] = 1.0
    params['seed'] = 0
  #  params['mode'] = 'uniform'

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', choices=['adult'], help='dataset to use')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--mode', choices=['off', 'dp', 'uniform', 'real'], help='prior distribution mode for PML')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data, workload = benchmarks.adult_benchmark()
    total = data.df.shape[0]

    measurements = privbayes_measurements(data, 1.0, args.seed, 'dp')
    est = privbayes_inference(data.domain, measurements, total=total)
   # est.df.to_csv('privbayes.csv')
    
    elim_order = [m[3][0] for m in measurements][::-1]

    #projections = [m[3] for m in measurements]
    
    #est2, _, _ = mechanism.run(data, projections, eps=args.epsilon/2, frequency=50, seed=args.seed, iters=args.iters) #

    # engine = FactoredInference(data.domain, iters=args.iters, warm_start=False, elim_order=elim_order)
    # est3 = engine.estimate(measurements, total=len(est.df))
   # synth3 = est3.synthetic_data(rows=len(est.df))
   # synth3.df.to_csv('privbayes_pgm.csv')

    measurements_uniform = privbayes_measurements(data, 1.0, args.seed, 'uniform')
    est4 = privbayes_inference(data.domain, measurements_uniform, total=total)
    # est4.df.to_csv('uniform.csv')
    # elim_order2 = [m[3][0] for m in measurements_uniform][::-1]
    # engine2 = FactoredInference(data.domain, iters=args.iters, warm_start=False, elim_order=elim_order2)
    # est6 = engine.estimate(measurements_uniform, total=len(est4.df))
    # synth6 = est6.synthetic_data(rows=len(est4.df))
    # synth6.df.to_csv('uniform_pgm.csv')

    # prior same as real data (!!!expermiental, possible privacy breach)
    measurements_real = privbayes_measurements(data, 1.0, args.seed, 'real')
    est5 = privbayes_inference(data.domain, measurements_real, total=total)
#     # est5.df.to_csv('real.csv')
#     # elim_order3 = [m[3][0] for m in measurements_real][::-1]
#     # engine3 = FactoredInference(data.domain, iters=args.iters, warm_start=False, elim_order=elim_order3)
#     # est7 = engine.estimate(measurements_real, total=len(est5.df))
#     # synth7 = est7.synthetic_data(rows=len(est5.df))
#     # synth7.df.to_csv('real_pgm.csv')

#     ### NO PRIVACY
#     # measurements_nop = privbayes_measurements(data, 1.0, args.seed, 'off')
#     # est_nop = privbayes_inference(data.domain, measurements_nop, total=total)
#     # est_nop.df.to_csv('no_privacy.csv')
    
#     # elim_order = [m[3][0] for m in measurements_nop][::-1]
#     # engine = FactoredInference(data.domain, iters=args.iters, warm_start=False, elim_order=elim_order)
#     # est_nop2 = engine.estimate(measurements_nop, total=len(est_nop.df))
#     # synth_nop = est_nop2.synthetic_data(rows=len(est_nop.df))
#     # synth_nop.df.to_csv('no_privacy_pgm.csv')


    def err(true, est):
        return np.sum(np.abs(true - est)) / true.sum()

    err_pb = []
    #err_pgm = []
    # err_pbpgm = []
    err_uni = []
    # err_uni_pgm = []
    err_real = []
#     # err_real_pgm = []
#     # err_nop = []
#     # err_nop_pgm = []
    

    for p, W in workload:
        true = W.dot(data.project(p).datavector())
        pb = W.dot(est.project(p).datavector())
     #   pgm = W.dot(est2.project(p).datavector())
        # pbpgm = W.dot(est3.project(p).datavector())
        uni = W.dot(est4.project(p).datavector())
#         # unipgm = W.dot(est6.project(p).datavector())
        real = W.dot(est5.project(p).datavector())
#         # realpgm = W.dot(est7.project(p).datavector())
#         # nop = W.dot(est_nop.project(p).datavector())
#         # nop_pgm = W.dot(est_nop2.project(p).datavector())
        err_pb.append(err(true, pb))
        #err_pgm.append(err(true, pgm))
        # err_pbpgm.append(err(true, pbpgm))
        err_uni.append(err(true, uni))
#         # err_uni_pgm.append(err(true, unipgm))
        err_real.append(err(true, real))
#         # err_real_pgm.append(err(true, realpgm))
#         # err_nop.append(err(true, nop))
#         # err_nop_pgm.append(err(true, nop_pgm))

#    # print('Not private models')
    print('Error of PrivBayes    : %.3f' % np.mean(err_pb))
   # print('Error of PrivBayes+PGM: %.3f' % np.mean(err_pgm))
    # print('Error of PrivBayes+PGM: %.3f' % np.mean(err_pbpgm))
    print('Error of PrivBayes with uniform prior: %.3f' % np.mean(err_uni))
#     # print('Error of PrivBayes+PGM with uniform prior: %.3f' % np.mean(err_uni_pgm))
    print('Error of PrivBayes with prior taken from real data: %.3f' % np.mean(err_real))
    # print('Error of PrivBayes+PGM with prior taken from real data: %.3f' % np.mean(err_real_pgm))
    # print('Error of PrivBayes with no privacy: %.3f' % np.mean(err_nop))
    # print('Error of PrivBayes+PGM with no privacy: %.3f' % np.mean(err_nop_pgm))

