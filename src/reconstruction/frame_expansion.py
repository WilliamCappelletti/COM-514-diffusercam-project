from numbers import Number
from typing import Tuple, Union

import numpy as np
from pycsou.core import DifferentiableMap
from pycsou.core.linop import LinearOperator


class FrameExpansionAdjoint(LinearOperator):
    def __init__(self, frame_shape, content_shape, content_selection) -> None:
        content_size = content_shape[0] * content_shape[1]
        frame_size = frame_shape[0] * frame_shape[1]

        super().__init__(shape=(content_size, frame_size), lipschitz_cst=1)

        self.frame_shape = frame_shape
        self.content_shape = content_shape
        self.content_selection = content_selection

    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return arg.reshape(self.frame_shape)[self.content_selection].ravel()

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        raise NotImplementedError()


class FrameExpansion(DifferentiableMap):
    """
    Places the contents of a vector, representing a 2D image, inside a bigger (2D) frame.

    dim: `(int, int)`
        The shape of the final frame

    margins: `((int, int), (int, int))`
        The intervals (on the X and Y axes of the bigger frame) in which to place the image, i.e.
        ((start_X, end_X), (start_Y, end_Y)). Note that endpoints are NOT included
    """

    def __init__(self, dim: Tuple[int, int], margins: Tuple[Tuple[int, int], Tuple[int, int]]):
        size = dim[0] * dim[1]

        (x_start, x_end) = margins[0]
        (y_start, y_end) = margins[1]

        # Shape of the content
        self.content_shape = (x_end - x_start, y_end - y_start)

        super().__init__(
            shape=(size, self.content_shape[0] * self.content_shape[1]),
            is_linear=True,
            lipschitz_cst=1,
            diff_lipschitz_cst=1,
        )

        self.frame_shape = dim
        self.content_selection = (slice(x_start, x_end), slice(y_start, y_end))
        self.padding = ((x_start, dim[0] - x_end), (y_start, dim[1] - y_end))

        self.adjointOperator = FrameExpansionAdjoint(
            frame_shape=self.frame_shape,
            content_shape=self.content_shape,
            content_selection=self.content_selection,
        )

    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return np.pad(arg.reshape(self.content_shape), self.padding).ravel()

    def jacobianT(self, arg: Union[Number, np.ndarray, None] = None) -> "LinearOperator":
        return self.adjointOperator
