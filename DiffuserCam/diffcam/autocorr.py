import numpy as np

import scipy
import scipy.fft as fft
from scipy.ndimage import fourier_shift

def autocorr2d(vals, pad_mode="reflect"):
    """
    Compute 2-D autocorrelation of image via the FFT.

    Parameters
    ----------
    vals : py:class:`~numpy.ndarray`
        2-D image.
    pad_mode : str
        Desired padding. See NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Return
    ------
    autocorr : py:class:`~numpy.ndarray`
    """

    (x_dim, y_dim) = vals.shape
    (x_padding, y_padding) = (x_dim//2, y_dim//2)

    padded_signal = np.pad(vals, ((x_padding, x_padding), (y_padding, y_padding)), pad_mode)
    
    # Compute DFT of padded signal
    dft_signal = fft.fft2(padded_signal)
    
    # Convolve it (with conjugate & time-reversed signal)
    dft_convolution = dft_signal * np.conjugate(dft_signal)
    # and center result (i.e. prepare top-left corner for cropping)
    dft_convolution = fourier_shift(dft_convolution, shift=(x_padding, y_padding), axis=-1)

    # Recover time-domain signal & crop it to original size
    obtained_signal = fft.ifft2(dft_convolution)[:x_dim, :y_dim]

    # Discard imaginary part (computation artifact)
    return np.real(obtained_signal)
