import numpy as np
from pycsou.core import LinearOperator
from scipy.fft import dctn, idctn


class DCTTransform(LinearOperator):
    def __init__(
        self,
        size: int,
        shape: tuple,
        type_: int = 2,
        workers: int = -1,
        norm: str = "ortho",
        axes: int = [0, 1],
        dtype: type = np.float64,
    ):
        self.type_ = type_
        self.workers = workers
        self.norm = norm
        self.axes = axes
        self.data_shape = shape
        self.dtype = dtype
        super(DCTTransform, self).__init__(shape=(size, size))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return dctn(
            x.reshape(self.data_shape),
            type=self.type_,
            axes=self.axes,
            norm=self.norm,
            workers=self.workers,
        ).flatten()

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return idctn(
            y.reshape(self.data_shape),
            type=self.type_,
            axes=self.axes,
            norm=self.norm,
            workers=self.workers,
        ).flatten()


def padding(x, data_pad, pad_vals):
    if data_pad[0]:
        x = np.pad(x, ((pad_vals[0], pad_vals[0]), (0, 0)), "constant")
    if data_pad[1]:
        x = np.pad(x, ((0, 0), (pad_vals[1], pad_vals[1])), "constant")
    return x
