from typing import Callable, Dict

from pycsou.func.loss import SquaredL2Loss
from pycsou.linop.conv import Convolve2D

from src.reconstruction.convolution_fft import Convolve2D_fft
from src.reconstruction.frame_expansion import FrameExpansion
from src.reconstruction.regularizations import available_regs

Regularization = Callable[[Callable], Dict]


def setup_loss(data, psf):
    Conv = Convolve2D_fft(size=data.size, filter=psf, shape=data.shape)
    Conv.compute_lipschitz_cst()

    data_vector = data.ravel()

    Loss = (1 / 2) * SquaredL2Loss(dim=data_vector.size, data=data_vector)
    Loss.compute_lipschitz_cst()

    return {"Loss": Loss, "M": Conv}


# Dispatch method for regularization
def setup_regularization(reg_name: str, config: Dict) -> Dict:
    reg_func = available_regs[reg_name][0]
    config.update(reg_func(**{k: v for k, v in config.items() if v is not None}))
    return config


# Dispatch method for optimizer
def optimize(reg_name, config: Dict) -> Dict:
    opti_func = available_regs[reg_name][1]
    config.update(opti_func(**{k: v for k, v in config.items() if v is not None}))
    rec_func = available_regs[reg_name][2]
    config.update(rec_func(**{k: v for k, v in config.items() if v is not None}))
    return config


def compute_loss(config: Dict) -> float:
    reconstructed = config["reconstructed"]
    loss = config["Loss"]
    M = config["M"]

    # Detect use of FrameExpansion
    if "frame_expansion" in config:
        FrameExp = FrameExpansion(
            dim=config["data_shape"], margins=config["frame_expansion"]["content_margins"]
        )
        M_prime = M * FrameExp
        return (loss * M_prime)(reconstructed.ravel())
    else:
        return (loss * M)(reconstructed.ravel())
