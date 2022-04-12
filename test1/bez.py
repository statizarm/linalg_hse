import numpy as np
from numpy.typing import NDArray
from numpy import linalg as la
from math import comb


class Interpolator:
    def __init__(self, points: NDArray):
        assert len(points.shape) == 2

        self._n = points.shape[1]
        self._m = np.empty((points.shape[0], 0))

        for i in range(self._n):
            self._m = np.concatenate(
                (
                    self._m,
                    points[:, i] * comb(self._n - 1, i),
                ),
                axis=1,
            )

    def _t_vect(self, t):
        t_pow = np.arange(self._n)
        t_1_pow = np.arange(self._n - 1, -1, -1)
        return np.matrix(
            t ** t_pow * (1 - t) ** t_1_pow
        ).transpose()

    def __call__(self, t):
        vect = self._t_vect(t)
        res = np.matmul(self._m, self._t_vect(t))
        return res

    def __str__(self):
        return '\n'.join(
            [
                '\t'.join(
                    [
                        f'{self._m[i, j]}*(1-t)^{self._n - 1 - j}*t^{j}'
                        for j in range(self._m.shape[1])
                    ]
                ) for i in range(self._m.shape[0])
            ]
        )
