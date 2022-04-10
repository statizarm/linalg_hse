from functools import wraps
import numpy as np
from numpy import linalg as la


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


def gauss(A: np.matrix, eps=1e-6):
    B = A.copy()
    n_rows, n_cols = B.shape
    permut = list(range(n_rows))
    for i in range(min(n_rows, n_cols)):
        for j in range(i, n_rows):
            if abs(B[j, i]) > eps:
                break
        else:
            continue
        B[[i, j]] = B[[j, i]]
        permut[i], permut[j] = permut[j], permut[i]

        B[i] = B[i] / B[i, i]

        for j in range(0, n_rows):
            if j == i: 
                continue
            B[j] = B[j] - (B[j, i] / B[i, i]) * B[i]

    return B, permut


def decomposition(A: np.matrix, eps=1e-6):
    B, permut = gauss(A, eps)
    n_rows, n_cols = B.shape
    for i in range(n_rows):
        for j in range(0, n_cols):
            if abs(B[i, j]) > eps:
                break
        else:
            break

    F = A[permut]
    F = F[:, :i]
    G = B[:i]

    return F, G


def pseudo_inv(A):
    F, G = decomposition(A)
    print('F', F)
    print('G', G)

    G_plus = np.matmul(G.transpose(), la.inv(np.matmul(G, G.transpose())))
    F_plus = np.matmul(la.inv(np.matmul(F.transpose(), F)), F.transpose())
    A_plus = np.matmul(G_plus, F_plus)

    return A_plus


@task(1)
def first_task():
    A = np.matrix(
        [
            [4., -15., 15.,],
            [-2., 9., -3.,],
            [-1., 5., 0.,],
            [3., -11., 12.,],
        ]
    )

    return pseudo_inv(A)


@task(2)
def second_task():
    A = np.matrix(
        [
            [-2., 8., 14., -12.],
            [-1., 5., 11., -3.],
            [4., -10., 13., -9.],
            [3., -7., 10., 0],
        ]
    )

    B = np.matrix(
        [
            [2.],
            [1.],
            [6.],
            [6.],
        ]
    )

    A_plus = pseudo_inv(A)
    print('A+', A_plus)

    return np.matmul(A_plus, B)


def main():
    for t in TASKS:
        t()


if __name__ == '__main__':
    main()
