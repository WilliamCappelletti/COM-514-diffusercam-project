from numbers import Number
from typing import Union

import numpy as np
from pycsou.core.functional import DifferentiableFunctional


class HuberNorm(DifferentiableFunctional):

    M: int = 0
    delta: float = None

    def __init__(self, M: int, delta: float):
        super().__init__(M, is_linear=False, lipschitz_cst=delta, diff_lipschitz_cst=1)
        self.M = M
        self.delta = delta

    def __get_cases_mask(self, x):
        """
        Returns two (complementary) masks, indicating whether each coordinate `x_i` satisfies:
            - `|x_i| <= delta`  or
            - `|x_i| > delta`
        """
        small_coords_mask = np.abs(x) <= self.delta

        return (small_coords_mask, np.logical_not(small_coords_mask))

    def __huber(self, x: np.ndarray):
        (small_coords_mask, big_coords_mask) = self.__get_cases_mask(x)

        # Summing 'small' coordinates: Huber function 1/2 * z^2
        small_coords_value = ((1 / 2) * np.square(x[small_coords_mask])).sum()

        # Summing 'big' coordinates: Huber function delta * (|z| - delta/2)
        big_coords_value = (self.delta * (np.abs(x[big_coords_mask]) - self.delta / 2)).sum()

        return small_coords_value + big_coords_value

    def __dim_check(self, data):
        if data.shape != (self.M,) and data.shape != (self.M, 1):
            raise ValueError("dimension mismatch")

    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        # Allow shortcut (scalar) notation if `dim==1`
        if isinstance(arg, Number):
            x = np.array([arg])
        else:
            x = arg

        self.__dim_check(arg)

        vals = self.__huber(x)

        return vals

    def __huber_gradient(self, x: np.ndarray):
        (small_coords_mask, big_coords_mask) = self.__get_cases_mask(x)

        # Leverage semantics of multiplication by `bool` array to keep the right size
        small_coords_values = small_coords_mask * x

        big_coords_values = big_coords_mask * (self.delta * np.sign(x))

        return small_coords_values + big_coords_values

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        # Allow shortcut (scalar) notation if `dim==1`
        if isinstance(arg, Number):
            x = np.array([arg])
        else:
            x = arg

        self.__dim_check(arg)

        gradient = self.__huber_gradient(x)

        if isinstance(arg, Number):
            return gradient[0]
        else:
            return gradient
