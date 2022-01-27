import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, minimize_scalar

T = 48

# Customer's utility function: for optimizing wrt b_pm
def u_b(b_pm, p, l, c, e):
    """
    b_pm: charging profile (1x2T)
    p: price vector (1xT)
    b_m: discharging profile (1xT)
    l: load profile (1xT)
    c: running cost of storage
    e: storage capacity 
    """
    return np.sum(p * (b_pm[:T] - b_pm[T:] + l)) + c*e


# Customer's utility function: for optimizing wrt e
def u_e(e, b_pm, p, l, c):
    """
    b_pm: charging profile (1x2T)
    p: price vector (1xT)
    b_m: discharging profile (1xT)
    l: load profile (1xT)
    c: running cost of storage
    e: storage capacity 
    """
    return np.sum(p * (b_pm[:T] - b_pm[T:] + l)) + c*e

def optimize_e(n, b_pm, p, l, c, e0):
    """
    n: customer's index
    """
    # generate constraint
    constr_ev = np.zeros(T)
    for i in range(T):
        constr_ev[i] = b_pm[n, i] + e0[n] + np.sum(b_pm[n, :i] - b_pm[n, T : T+i])
    constr_e = np.max(constr_ev)

    e_opt = minimize_scalar(u_e, args=(b_pm[n, :], p, l[n, :], c[n]),
        method='Bounded',
        bounds=(constr_e, 5.))
    return e_opt.x

def constr_matrix_b(alpha):
    """
    alpha: efficiency
    """
    # storage efficiency
    mat1 = np.zeros(shape=(1, 2*T), dtype='float')
    for j in range(T):
        mat1[0, j] = -alpha
        mat1[0, j+T] = 1

    # within charging and discharging capacity
    mat2 = np.hstack([np.identity(T, dtype='float'), np.zeros(shape=(T, T), dtype='float')])
    mat3 = np.hstack([np.zeros(shape=(T, T), dtype='float'), np.identity(T, dtype='float')])

    # energy that can be stored or used at a time slot
    mat4 = np.zeros(shape=(T, 2*T), dtype='float')
    for i in range(1, T):
        mat4[i, :i] = -alpha
        mat4[i, T:T+i] = alpha
    for i in range(T):
        mat4[i, T+i] = 1

    mat5 = np.zeros(shape=(T, 2*T), dtype='float')
    for i in range(1, T):
        mat5[i, :i] = 1
        mat5[i, T:T+i] = -1
    for i in range(T):
        mat5[i, i] = 1    

    return np.vstack([mat1, mat2, mat3, mat4, mat5, mat3])

def constr_vectors_b(n, b_sup, b_inf, alpha, e0, e, l):
    """
    n: customer's index
    """
    # left
    leftv1 = np.zeros(1)          # bilateral
    leftv2 = np.zeros(T)          # bilateral
    leftv3 = np.zeros(T)          # bilateral
    leftv4 = -np.ones(T)*np.inf   # unilateral
    leftv5 = -np.ones(T)*np.inf   # unilateral
    leftv_constr = np.concatenate([leftv1, leftv2, leftv3, leftv4, leftv5, leftv3])

    # right
    rightv_constr = np.concatenate([[0], [b_sup] * T, [b_inf] * T, [alpha * e0[0]] * T, [e[0] - e0[0]] * T, l[0, :]])
    return leftv_constr, rightv_constr

def optimize_b(n, b_sup, b_inf, p, c, alpha, e0, e, l):
    """
    n: customer's index
    """
    mat_constr = constr_matrix_b(alpha)
    leftv_constr, rightv_constr = constr_vectors_b(n, b_sup, b_inf, alpha, e0, e, l)
    linear_constraint = LinearConstraint(mat_constr, leftv_constr, rightv_constr)
    bounds = Bounds([0] * 2 * T, [max(b_sup, b_inf)] * 2 * T)
    x0 = np.zeros(2 * T)
    b_opt = minimize(u_b, x0, args=(p, l[n, :], c[n], e[n]), method='trust-constr',
        constraints=linear_constraint,
        bounds=bounds,
        #options={'factorization_method' : 'SVDFactorization', 'verbose' : 0})
        options={'verbose' : 0})
    return b_opt.x