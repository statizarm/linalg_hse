import numpy as np
from numpy import linalg as la
from numpy.typing import (
    NDArray,
)
from itertools import (
    combinations,
)

from functools import (
    reduce,
)


class Interpolator:
    def __init__(self, x: NDArray, y: NDArray):
        assert len(x.shape) == 1 and len(y.shape) == 1
        self._n, = x.shape
        self._vand = np.zeros((self._n, self._n))

        for i in range(self._n):
            self._vand[i] = x[i] ** np.arange(self._n)

        self._y = y.copy()
        self._det = la.det(self._vand)

    def __call__(self, x: float):
        res = 0
        for i in range(self._n):
            vand = self._vand.copy()
            vand[i] = x ** np.arange(self._n)
            res += self._y[i] * la.det(vand) / self._det

        return res

    def _calc_coefs(self):
        xs = np.array([self._vand[i, 1] for i in range(self._n)])
        res = np.zeros((self._n,))
        for i in range(self._n):
            xs_ = np.delete(xs, i)
            div = np.prod(xs[i] - xs_)
            l = np.array(
                [
                    sum(
                        reduce(
                            lambda x, y: (-1) ** j * x * y, comb, 1,
                        ) for comb in combinations(xs_, j)
                    ) for j in range(res.shape[0])
                ]
            )
            res += self._y[i] * l / div
        return res

    @property
    def coefs(self):
        if getattr(self, '_coefs', None) is None:
            setattr(self, '_coefs', self._calc_coefs())

        return self._coefs

    def calc(self, x):
        coefs = self.coefs
        xs = x ** np.arange(self._n - 1, -1 , -1)

        return np.sum(coefs * (xs))

    
    def __str__(self):
        return ' + '.join(
            [
                f'{round(k, 2)}*x^{self._n - i - 1}' for i, k in enumerate(self.coefs)
            ]
        )
