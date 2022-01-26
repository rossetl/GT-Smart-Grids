import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import poisson
from utils import optimize_e, optimize_b
import sys
import multiprocessing as mp

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

pool = mp.Pool(mp.cpu_count())
e_opt_dict = {}
b_opt_dict = {}
for d in range(par['n_days']):
    profiles = np.ndarray((par['n_nodes'], T))

    # optimize e
    for n in range(par['n_nodes']):
        e_opt_dict[n] = pool.apply_async(optimize_e, args=(n, b_pm, p, l, c, e0))
    e_opt = np.array([e_opt_dict[n].get(timeout=180) for n in range(par['n_nodes'])])

    # update e
    e = e + par['beta_1']*(e_opt - e)

    # optimize b
    for n in range(par['n_nodes']):
        b_opt_dict[n] = pool.apply_async(optimize_b, args=(n, par['b_sup'], par['b_inf'], p, c, par['alpha'], e0, e, l))
    b_opt = np.array([b_opt_dict[n].get(timeout=180) for n in range(par['n_nodes'])])

    # update b
    b_pm = b_pm + par['beta_2'] * (b_opt - b_pm)

    # update e0
    e0 = e0 + b_pm[:, :T].sum() - b_pm[:, T:].sum()

    b_history = np.append(b_history, [np.mean(b_pm[:, :T] - b_pm[:, T:], axis=0)], axis=0)
    p = market_price(l.sum(axis=0) - b_pm[:, T:].sum(axis=0) + b_pm[:, :T].sum(axis=0))
    p_history = np.append(p_history, [p], axis=0)
    e_history = np.append(e_history, e.sum())
    print('completed day {d}/{n_days}'.format(d=d+1, n_days=par['n_days']))
pool.close()

np.save('Results/storage_profile.npy', b_history)
np.save('Results/market_prices.npy', p_history)
np.save('Results/storage.npy', e_history)

print('simulation completed!')