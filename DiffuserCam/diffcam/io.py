import os.path

import cv2
import numpy as np
import rawpy
from diffcam.constants import RPI_HQ_CAMERA_BLACK_LEVEL, RPI_HQ_CAMERA_CCM_MATRIX
from diffcam.plot import plot_image
from diffcam.util import bayer2rgb, print_image_info, resize, rgb2gray


def load_image(
    fp,
    verbose=False,
    flip=False,
    bayer=False,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    blue_gain=None,
    red_gain=None,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    back=None,
    nbits_out=None,
):
    """
    Load image as numpy array.

    Parameters
    ----------
    fp : str
        Full path to file.
    verbose : bool, optional
        Whether to plot into about file.
    flip : bool
    bayer : bool
    blue_gain : float
    red_gain : float

    Returns
    -------
    img :py:class:`~numpy.ndarray`
        RGB image of dimension (height, width, 3).
    """
    assert os.path.isfile(fp)
    if "dng" in fp:
        assert bayer
        raw = rawpy.imread(fp)
        img = raw.raw_image
        # TODO : use raw.postprocess?
        ccm = raw.color_matrix[:, :3]
        black_level = np.array(raw.black_level_per_channel[:3]).astype(np.float32)
    else:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

    if bayer:
        assert len(img.shape) == 2, img.shape
        if img.max() > 255:
            # HQ camera
            n_bits = 12
        else:
            n_bits = 8

        if back:
            back_img = cv2.imread(back, cv2.IMREAD_UNCHANGED)
            dtype = img.dtype
            img = img.astype(np.float32) - back_img.astype(np.float32)
            img = np.clip(img, a_min=0, a_max=img.max())
            img = img.astype(dtype)
        if nbits_out is None:
            nbits_out = n_bits
        img = bayer2rgb(
            img,
            nbits=n_bits,
            bg=blue_gain,
            rg=red_gain,
            black_level=black_level,
            ccm=ccm,
            nbits_out=nbits_out,
        )

    else:
        assert len(img.shape) == 3
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if flip:
        img = np.flipud(img)
        img = np.fliplr(img)

    if verbose:
        print_image_info(img)

    return img


def load_psf(
    fp,
    downsample=1,
    return_float=True,
    bg_pix=(5, 25),
    return_bg=False,
    flip=False,
    verbose=False,
    bayer=False,
    blue_gain=None,
    red_gain=None,
    dtype=np.float32,
    nbits_out=None,
    single_psf=False,
):
    """
    Load and process PSF for analysis or for reconstruction.

    Basic steps are:
    - Load image.
    - (Optionally) subtract background. Recommended.
    - (Optionally) resize to more manageable size
    - (Optionally) normalize within [0, 1] if using for reconstruction; otherwise cast back to uint for analysis.

    Parameters
    ----------
    fp : str
        Full path to file.
    downsample : int, optional
        Downsampling factor. Recommended for image reconstruction.
    return_float : bool, optional
        Whether to return PSF as float array, or unsigned int.
    bg_pix : tuple, optional
        Section of pixels to take from top left corner to remove background level. Set to `None` to omit this
        step, althrough it is highly recommended.
    return_bg : bool, optional
        Whether to return background level, for removing from data for reconstruction.
    flip : bool, optional
        Whether to flip up-down and left-right.
    verbose

    Returns
    -------
    psf :py:class:`~numpy.ndarray`
        2-D array of PSF.
    """

    # load image data and extract necessary channels
    psf = load_image(
        fp,
        verbose=verbose,
        flip=flip,
        bayer=bayer,
        blue_gain=blue_gain,
        red_gain=red_gain,
        nbits_out=nbits_out,
    )

    original_dtype = psf.dtype
    psf = np.array(psf, dtype=dtype)

    # subtract background, assume black edges
    bg = np.zeros(3)
    if bg_pix is not None:
        bg = []
        for i in range(3):
            bg_i = np.mean(psf[bg_pix[0] : bg_pix[1], bg_pix[0] : bg_pix[1], i])
            psf[:, :, i] -= bg_i
            bg.append(bg_i)
        psf = np.clip(psf, a_min=0, a_max=psf.max())
        bg = np.array(bg)

    # resize
    if downsample != 1:
        psf = resize(psf, 1 / downsample)

    if single_psf:
        assert len(psf.shape) == 3
        # TODO : in Lensless Learning, they sum channels --> `psf_diffuser = np.sum(psf_diffuser,2)`
        # https://github.com/Waller-Lab/LenslessLearning/blob/master/pre-trained%20reconstructions.ipynb
        psf = np.sum(psf, 2)
        psf = psf[:, :, np.newaxis]

    # normalize
    if return_float:
        # psf /= psf.max()
        psf /= np.linalg.norm(psf.ravel())
    else:
        psf = psf.astype(original_dtype)

    if return_bg:
        return psf, bg
    else:
        return psf


def load_data(
    psf_fp,
    data_fp,
    downsample,
    data_truth_fp=None,
    bg_pix=(5, 25),
    plot=True,
    flip=False,
    bayer=False,
    blue_gain=None,
    red_gain=None,
    gamma=None,
    gray=False,
    dtype=np.float32,
    single_psf=False,
):
    """
    Load data for image reconstruction.

    Parameters
    ----------
    psf_fp : str
        Full path to PSF file.
    data_fp : str
        Full path to measurement file.
    data_truth_fp : str
        Full path to ground truth image file
    source : "white", "red", "green", or "blue"
        Light source used to measure PSF.
    downsample : int or float
        Downsampling factor.
    bg_pix : tuple, optional
        Section of pixels to take from top left corner to remove background
        level. Set to `None` to omit this step, although it is highly
        recommended.
    plot : bool, optional
        Whether or not to plot PSF and raw data.
    flip : bool, optional
        Whether to flip data.
    cv : bool, optional
        Whether image was saved with OpenCV. If not colors need to be swapped.

    Returns
    -------
    psf :py:class:`~numpy.ndarray`
        2-D array of PSF.
    data :py:class:`~numpy.ndarray`
        2-D array of raw measurement data.
    truth_img :py:class:`~numpy.ndarray`
        2-D array of ground truth image
    """

    assert os.path.isfile(psf_fp)
    assert os.path.isfile(data_fp)
    if data_truth_fp:
        assert os.path.isfile(data_truth_fp)

    # load and process PSF data
    psf, bg = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        bg_pix=bg_pix,
        return_bg=True,
        flip=flip,
        bayer=bayer,
        blue_gain=blue_gain,
        red_gain=red_gain,
        dtype=dtype,
        single_psf=single_psf,
    )

    # load and process raw measurement
    data = process_image(
        data_fp, flip, bayer, blue_gain, red_gain, dtype, bg, psf.shape, downsample
    )
    if data_truth_fp:
        truth_img = process_image(
            data_truth_fp, flip, bayer, blue_gain, red_gain, dtype, bg, psf.shape, downsample
        )
    else:
        truth_img = None

    if gray:
        psf = rgb2gray(psf)
        data = rgb2gray(data)
        if data_truth_fp:
            truth_img = rgb2gray(truth_img)

    if plot:
        ax = plot_image(psf, gamma=gamma)
        ax.set_title("PSF")
        ax = plot_image(data, gamma=gamma)
        ax.set_title("Raw data")
        if data_truth_fp:
            ax = plot_image(truth_img, gamma=gamma)
            ax.set_title("Ground truth data")

    return psf, data, truth_img


def process_image(img_fp, flip, bayer, blue_gain, red_gain, dtype, bg, psf_shape, downsample):
    data = load_image(img_fp, flip=flip, bayer=bayer, blue_gain=blue_gain, red_gain=red_gain)
    data = np.array(data, dtype=dtype)

    data -= bg
    data = np.clip(data, a_min=0, a_max=data.max())
    if data.shape != psf_shape:
        # in DiffuserCam dataset, images are already reshaped
        data = resize(data, 1 / downsample)
    data /= np.linalg.norm(data.ravel())
    return data
