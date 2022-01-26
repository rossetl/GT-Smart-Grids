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
    mat4[0, T] = 1
    for i in range(1, T):
        for j in range(i):
            mat4[i, j] = -alpha
            mat4[i, j+T] = alpha
        mat4[i-1, i-1+T] = 1

    mat5 = np.zeros(shape=(T, 2*T), dtype='float')
    mat5[0, 0] = 1
    for i in range(1, T):
        for j in range(i):
            mat5[i, j] = 1
            mat5[i, j+T] = -1
        mat5[i-1, i-1] = 1    

    return np.vstack([mat1, mat2, mat3, mat4, mat5, mat3])

def constr_vectors_b(n, b_sup, b_inf, alpha, e0, e, l):
    """
    n: customer's index
    """
    # left
    leftv_constr = - np.ones(241) * np.inf
    leftv_constr[0] = 0

    # right
    rightv_constr = np.concatenate([[0], [b_sup] * T, [b_inf] * T, [alpha * e0[n]] * T, [e[n] - e0[n]] * T, l[n, :]])
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
        options={'verbose' : 0})
    return b_opt.x