import numpy as np
import scipy.fft as fft
from pycsou.core import LinearOperator

class Convolve2D_fft(LinearOperator):
    def __init__(self, size, filter, shape, dtype="float64"):

        dims = shape

        self.h = filter
        self.nh = np.array(self.h.shape)
        self.dirs = np.arange(len(dims))

        # padding
        if (filter.shape[0] % 2) == 0:
            offset0 = filter.shape[0] // 2 - 1
        else:
            offset0 = filter.shape[0] // 2
        if (filter.shape[1] % 2) == 0:
            offset1 = filter.shape[1] // 2 - 1
        else:
            offset1 = filter.shape[1] // 2

        offset = np.array((offset0, offset1), dtype=int)
        self.offset = 2 * (self.nh // 2 - offset)
        pad = [(0, 0) for _ in range(self.h.ndim)]
        dopad = False
        for inh, nh in enumerate(self.nh):
            if nh % 2 == 0:
                self.offset[inh] -= 1
            if self.offset[inh] != 0:
                pad[inh] = [
                    self.offset[inh] if self.offset[inh] > 0 else 0,
                    -self.offset[inh] if self.offset[inh] < 0 else 0,
                ]
                dopad = True
        if dopad:
            self.h = np.pad(self.h, pad, mode="constant")
        self.nh = self.h.shape

        # find out which directions are used for convolution and define offsets
        if len(dims) != len(self.nh):
            dimsh = np.ones(len(dims), dtype=np.int)
            for idir, dir in enumerate(self.dirs):
                dimsh[dir] = self.nh[idir]
            self.h = self.h.reshape(dimsh)

        if np.prod(dims) != size:
            raise ValueError("product of dims must equal N!")
        else:
            self.dims = np.array(dims)

        self.axes = [0, 1]
        self.intern_shape = np.array(filter.shape) + np.array(dims)
        # Precomputing dfts
        reverse = (slice(None, None, -1),) * len(dims)
        self.h_dft_rev_conj = fft.rfft2(self.h[reverse].conj(), self.intern_shape, axes=self.axes)
        self.h_dft = fft.rfft2(self.h, self.intern_shape, axes=self.axes)

        # Centering
        startind = (self.intern_shape - np.array(dims)) // 2
        endind = startind + np.array(dims)
        self.slice = tuple([slice(startind[k], endind[k]) for k in range(len(endind))])

        self.shape = (size, size)
        self.dtype = np.dtype(dtype)

        super(Convolve2D_fft, self).__init__(shape=self.shape)


    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.reshape(x, self.dims)
        y = self.fftconvolve(x, self.h_dft)
        y = y.ravel()
        return y

    def adjoint(self, x: np.ndarray) -> np.ndarray:
        x = np.reshape(x, self.dims)
        y = self.fftconvolve(x, self.h_dft_rev_conj)
        y = y.ravel()
        return y

    def fftconvolve(self, x, sp):
        x_dft = fft.rfft2(x, self.intern_shape, axes=self.axes)
        ret = fft.irfft2(x_dft * sp, self.intern_shape, axes=self.axes)
        return np.real(ret[self.slice])