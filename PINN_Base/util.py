from scipy.io import loadmat
import numpy as np
import sys


def bounds_from_data(X):
    lower_bound = np.min(X, axis=0)
    upper_bound = np.max(X, axis=0)

    return lower_bound, upper_bound


def random_choice(X, size=10000):
    idx = np.random.choice(list(range(X.shape[0])), size=size)
    return X[idx, :]


def random_choices(*args, size=10000):
    span = args[0].shape[0]
    idx = np.random.choice(list(range(span)), size=size)

    vals = []
    for X in args:
        assert(X.shape[0] == span)
        vals.append(X[idx, :])

    if len(vals) == 1:
        return vals[0]
    else:
        return vals


def percent_noise(U, noise_percent):
    std = np.std(U[:, 0]) * noise_percent
    return U + np.random.normal(0, std, size=U.shape)


def load_mat_2d(filename, x1_key, x2_key, u_key):
    data = loadmat(filename)

    x1 = data[x1_key]
    x2 = data[x2_key]
    u = np.real(data[u_key])

    return x1, x2, u


def ungrid_u_2d(x1, x2, u):
    n_total = x1.shape[0] * x2.shape[0]

    X = np.zeros((n_total, 2))
    U = np.zeros((n_total, 1))

    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            idx = i * x2.shape[0] + j
            X[idx, 0] = x1[i]
            X[idx, 1] = x2[j]
            U[idx, 0] = u[i, j]

    return X, U


def get_u_const_idx_2d(x1, x2, u, idx_1=None, idx_2=None):

    if idx_1 is not None:
        U = np.zeros((x2.shape[0], 1))
        X = np.zeros((x2.shape[0], 2))
        i = idx_1
        for j in range(x2.shape[0]):
            U[j, 0] = u[i, j]
            X[j, 0], X[j, 1] = x1[i], x2[j]

        return X, U
    elif idx_2 is not None:
        U = np.zeros((x1.shape[0], 1))
        X = np.zeros((x1.shape[0], 2))
        j = idx_2
        for i in range(x1.shape[0]):
            U[i, 0] = u[i, j]
            X[i, 0], X[i, 1] = x1[i], x2[j]

        return X, U
    else:
        return None


def to_grid_u_2d(nx1, nx2, U):

    u = np.zeros((nx1, nx2))

    for i in range(nx1):
        for j in range(nx2):
            idx = i * nx2 + j

            u[i, j] = U[idx, 0]

    return u


def make_fetches_callback():

    array = []

    def fetches_callback(*args):
        array.append(args)

    return array, fetches_callback


def bfgs_callback(loss):
    print(f"loss={loss:.2E}", end="\r")


def rmse(U_true, U_hat):
    return np.sqrt(np.mean((U_true[:, 0] - U_hat[:, 0])**2))


def rel_error(U_true, U_hat):
    return np.linalg.norm(U_true - U_hat, 2) / np.linalg.norm(U_true, 2)


def unwrap(w0):
    flats = []

    for l in w0:
        flats.append(l.flatten())

    return np.hstack(flats)
