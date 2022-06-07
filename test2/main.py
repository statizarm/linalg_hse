from functools import wraps
import numpy as np
from numpy import linalg as la
from scipy import optimize
import matplotlib.pyplot as plt

from pagerank import PageRank


TASKS = []


def task(number: int):
    def wrapper(foo):
        @wraps(foo)
        def real_wrapper(*args, **kwargs):
            print(f'=====================Task {number}=====================')
            res = foo(*args, **kwargs)
            print('Ans:', res)
            return res
        TASKS.append(real_wrapper)
        return real_wrapper
    return wrapper


@task(1)
def first_task():
    A = np.matrix(
        [
            [-48., -76., 24., -89., ],
            [30., -47., 66., 20., ],
            [-33., 58., -78., 56., ],
        ]
    )
    A_star = A.transpose()
    C = np.matmul(A_star, A)
    v, W = la.eigh(C, UPLO='L')
    v = np.flip(v)
    W = np.flip(W, axis=1)
    V_star = W.transpose()
    sigm = np.sqrt(v)
    print("v = ", v)
    print("w = ", W)
    print("sigm = ", sigm)
    print("V_star = ", V_star)
    U = np.zeros((3, 3))
    for i in range(3):
        if sigm[i] > 1e-4:
            U[:, i] = (1 / sigm[i] * np.matmul(A, W[:, i])).ravel()

    print("U = ", U)
    Z = np.eye(3, 4)
    print("Z = ", Z)
    for i in range(3):
        if i >= 2:
            Z[i, i] = 0
        else:
            Z[i, i] = sigm[i]

    print("Z_dot = ", Z)
    A1 = np.matmul(np.matmul(U, Z), V_star)
    print("A1 = ", A1)

    return la.norm(A - A1)


@task(2)
def second_task():
    A = np.matrix(
        [
            [1., .0,],
            [-1., -9.,],
        ]
    )

    b = np.matrix(
        [
            [1.],
            [-10.],
        ]
    )

    A_delta = np.matrix(
        [
            [.09, 0.02, ],
            [-.01, .01, ],
        ]
    )

    b_delta = np.matrix(
        [
            [.11, ],
            [.1, ],
        ]
    )

    A_delta_norm = la.norm(A_delta, ord=1)
    A_norm = la.norm(A, ord=1)
    b_delta_norm = la.norm(b_delta, ord=1)
    b_norm = la.norm(b + b_delta, ord=1)
    ae = A_norm * la.norm(la.inv(A), ord=1)

    delta_A = A_delta_norm / A_norm
    delta_b = b_delta_norm / b_norm

    dx = ae * (delta_A + delta_b)

    print('ae = ', ae)
    print('dA = ', delta_A)
    print('db = ', delta_b)
    print('dx <= ', dx)

    return dx


@task(3)
def third_task():
    A = np.matrix(
        [
            [6., -4.,],
            [6., -7.,],
        ]
    )

    A_delta = np.matrix(
        [
            [.01, .01, ],
            [.01, .01, ],
        ]
    )

    A_delta_norm = la.norm(A_delta, ord=1)
    A_norm = la.norm(A, ord=1)
    ae = A_norm * la.norm(la.inv(A), ord=1)
    delta_A = A_delta_norm / A_norm
    delta_A_inv = ae * delta_A / (1 - ae * delta_A)
    A_inv = la.inv(A)

    print("A_inv = ", A_inv)
    print("ae = ", ae)
    print("delta_A = ", delta_A)
    print("delta_A_inv = ", delta_A_inv)

    return delta_A_inv


@task(4)
def fourth_task():
    A = np.matrix(
        [
            [1, 0, 1, 1, 0,],
            [0, 0, 1, 0, 0,],
            [0, 0, 1, 1, 0,],
            [0, 0, 0, 0, 1,],
            [0, 1, 1, 0, 0,],
        ]
    )
    pagerank = PageRank()
    v, M = pagerank(A)

    print("M = ", M)
    print("v = ", v)
    return np.argmax(v) + 1


@task(7)
def seventh_task():
    # Что-то более адекватное я сделать не успел
    c = np.matrix(
        [
            [28.0, 123.0, 20.0, ],
        ]
    )

    A = np.matrix(
        [
            [-6., 8., 5., ],
            [4., 7., -7., ],
        ]
    )

    b = np.matrix(
        [
            [2.0, ],
            [3.0, ],
        ]
    )

    return optimize.linprog(c, -A, -b)

@task(8)
def eighth_task():
    A = np.matrix(
        [
            [24., 3., 9.,],
            [5., 21., 7.,],
            [7., 2., 25.,],
        ]
    )

    b = np.matrix(
        [
            [3., ],
            [4., ],
            [9., ],
        ]
    )

    x = np.matrix(
        [
            [0., ],
            [0., ],
            [0., ],
        ]
    )

    print(b)
    Alpha = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i == j:
                Alpha[i, j] = 0;
            else:
                Alpha[i, j] = - A[i, j] / A[i, i]
        b[i,0] = b[i,0] / A[i, i]

    def norm(dx):
        n = la.norm(Alpha, ord=np.Inf)
        return n / (1 - n) * la.norm(dx, ord=np.Inf)

    d = 1e-2

    for i in range(1000):
        new_x = b + Alpha @ x
        n = norm(new_x - x)
        if n < d:
            x = new_x
            break
        x = new_x

    print("solve = ", A @ x)
    print("iteration = ", i)

    return x


def main():
    for t in TASKS:
        t()


if __name__ == '__main__':
    main()
