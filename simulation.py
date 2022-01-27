import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import poisson
from utils import optimize_e, optimize_b
import sys
from joblib import Parallel, delayed
import time

with open('parameters.txt') as f:
    raw_par = f.read()
par = json.loads(raw_par)

# Import the load profile
load_curve = np.loadtxt('data/load_agg_data.csv', delimiter=',', skiprows=1)[:, 2]

# Import market price (MW, Euro/MWh)
market_price_data = np.loadtxt('data/market_price_UK.csv', delimiter=',')
# rescaling of the total demand to fit the simulation
tot_demand = market_price_data[:, 0] * 2.6 - 1
market_price = interp1d(tot_demand, market_price_data[:, 1] / 1000, kind='cubic')

# initialization
T = 48
c = np.random.rand(par['n_nodes'])*0.05        # running cost of using storage (Euro/kWh)
e0 = np.ones(par['n_nodes'])*0.05              # storage level
e = np.ones(par['n_nodes'])*0.1                # storage capacity
b_pm = np.zeros(shape=(par['n_nodes'], 2*T))   # charging profiles
l = np.zeros(shape=(par['n_nodes'], T))        # load profiles
b_history = np.ndarray(shape=(0, T))           # storage profile history
p_history = np.ndarray(shape=(0, T))           # price history
e_history = np.array([])                       # total storage

# generate load profiles
sigma = 0.005
for i in range(par['n_nodes']):
    l[i, :] = load_curve[(poisson.rvs(mu=2, size=T) + np.arange(T)) % T] + np.random.randn(T)*sigma

p = market_price(l.sum(axis=0) - b_pm[:, T:].sum(axis=0) + b_pm[:, :T].sum(axis=0))
p_history = np.append(p_history, [p], axis=0)
e_history = np.append(e_history, e.sum())

for d in range(par['n_days']):
    t_start = time.time()
    profiles = np.ndarray((par['n_nodes'], T))
    b_opt = np.ndarray(shape=(0, 2*T))
    # optimize e
    with Parallel(n_jobs=par['n_nodes']) as parallel:
        e_opt = parallel(delayed(optimize_e)(n, b_pm, p, l, c, e0) for n in range(par['n_nodes']))
    e_opt = np.array(e_opt)

    # update e
    e = e + par['beta_1']*(e_opt - e)

    # optimize b
    with Parallel(n_jobs=par['n_nodes']) as parallel:
        b_opt_list = parallel(delayed(optimize_b)(n, par['b_sup'], par['b_inf'], p, c, par['alpha'], e0, e, l) for n in range(par['n_nodes']))
    for n in range(par['n_nodes']):
        b_opt = np.append(b_opt, [b_opt_list[n]], axis=0)

    # update b
    b_pm = b_pm + par['beta_2'] * (b_opt - b_pm)

    # update e0
    e0 = e0 + b_pm[:, :T].sum() - b_pm[:, T:].sum()

    b_history = np.append(b_history, [np.mean(b_pm[:, :T] - b_pm[:, T:], axis=0)], axis=0)
    p = market_price(l.sum(axis=0) - b_pm[:, T:].sum(axis=0) + b_pm[:, :T].sum(axis=0))
    p_history = np.append(p_history, [p], axis=0)
    e_history = np.append(e_history, e.sum())
    t_stop = time.time()
    time_elapsed = round((t_stop - t_start)/60, 2)
    print('completed day {d}/{n_days} in {mins} min'.format(d=d+1, n_days=par['n_days'], mins=time_elapsed))

np.save('Results/storage_profile.npy', b_history)
np.save('Results/market_prices.npy', p_history)
np.save('Results/storage.npy', e_history)

print('simulation completed!')