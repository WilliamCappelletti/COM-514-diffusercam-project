from typing import Callable, Dict

import numpy as np
from pycsou.func.penalty import L1Norm, NonNegativeOrthant, SquaredL2Norm
from pycsou.linop import FirstDerivative

from src.reconstruction.dct import DCTTransform
from src.reconstruction.frame_expansion import FrameExpansion
from src.reconstruction.hubernorm import HuberNorm
from src.reconstruction.optimizers import apgd_optim, pds_optim

"""
Regularization strategies SHOULD follow Pycsou-style naming conventions. The optimization problem is
    min_x  Loss(y, Mx) + F'(x) + G(x) + H(Kx)
           \========|========/
                    V
    min_x         F(x)         + G(x) + H(Kx)

where:
    - Loss  : is a convex, differentiable loss functional (i.e. L2 loss)
    - M     : is a differentiable forward operator
    - F'    : is a convex, differentiable operator
    - G     : is a convex, proximable operator
    - H     : is a convex, proximable operator
    - K     : is a linear operator

"""
Regularization = Callable[[Callable], Dict]


# An optimizer returns a Dict containing AT LEAST:
#   (1) 'has_converged': bool indicating convergence
#   (2) 'iters_num':     the number of iterations performed
#   (3) 'reconstructed': the reconstructed signal
Optimizer = Callable[[], Dict]


# An estimate recovery takes the output of the optimization problem
# and converts it into the estimate of the original image
EstimateRecovery = Callable[[np.ndarray], np.ndarray]


def trivial_estimate_recovery(reconstructed, **kwargs):
    """
    Default estimate recovery
    (useful when there is NO need for coordinate transformation)
    """
    return {"reconstructed": reconstructed}


def tikhonov_reg(Loss, M, data_size, lambda_, **kwargs):
    """
    The Tikhonov regularization consists of a differentiable penalty functional,
    which can therefore be optimized (via Gradient Descent) together with the loss functional
    """

    Tikhonov = lambda_ * SquaredL2Norm(dim=data_size)

    return {"F": Loss * M + Tikhonov}


def lasso_reg(Loss, M, data_size, lambda_, **kwargs):
    # Proximable part
    Lasso = lambda_ * L1Norm(dim=data_size)

    return {"F": Loss * M, "G": Lasso}


def non_negative_reg(Loss, M, data_size, lambda_, **kwargs):
    # Proximable part
    Regularizer = NonNegativeOrthant(dim=data_size)

    return {"F": Loss * M, "G": Regularizer}


Dct = None


def dct_lasso_reg(Loss, M, data_size, lambda_, **kwargs):
    G = lambda_ * L1Norm(dim=data_size)

    # Perform Inverse DCT, BEFORE calculating the loss
    global Dct
    Dct = DCTTransform(size=data_size, shape=kwargs["data_shape"])

    M_prime = M * Dct.get_adjointOp()
    M_prime.compute_lipschitz_cst()

    F = Loss * M_prime

    return {"F": F, "G": G}


def dct_estimate_recovery(reconstructed, **kwargs):
    """
    Invert DCT transform, to get back the actual image estimate
    """
    print("inverting dct")
    global Dct

    time_domain_signal = Dct.adjoint(reconstructed)

    return {"reconstructed": time_domain_signal}


def tv_non_negative_reg(Loss, M, data_size, lambda_, **kwargs):

    G = NonNegativeOrthant(dim=data_size)

    K = FirstDerivative(size=data_size, kind="forward")
    K.compute_lipschitz_cst(tol=1e-2)

    H = lambda_ * L1Norm(dim=data_size)

    return {"F": Loss * M, "G": G, "H": H, "K": K}


def huber_non_neg_reg(Loss, M, data_size, lambda_, **kwargs):
    print(f"Executing huber non negative, lambda={lambda_}")
    Derivative = FirstDerivative(size=data_size, kind="forward")
    Derivative.compute_lipschitz_cst(tol=1e-2)

    # This functional is differentiable
    F_prime = lambda_ * HuberNorm(M=data_size, delta=0.001) * Derivative
    # hence can be optimized using gradient descent
    F = Loss * M + F_prime

    G = NonNegativeOrthant(dim=data_size)

    return {"F": F, "G": G}


#########################################
#   Regularizations on cropped images   #
#########################################

content_margins = ((172, 415), (278, 710))
(x_start, x_end) = content_margins[0]
(y_start, y_end) = content_margins[1]
content_shape = (x_end - x_start, y_end - y_start)
content_size = content_shape[0] * content_shape[1]
frame_expansion_config = {
    "content_shape": content_shape,
    "content_size": content_size,
    "content_margins": content_margins,
}


def fe_lasso_reg(Loss, M, data_shape, lambda_, **kwargs):

    # Proximable part
    Lasso = lambda_ * L1Norm(dim=content_size)

    # Perform expansion to original size, before applying convolution
    M_prime = M * FrameExpansion(dim=data_shape, margins=content_margins)

    return {"F": Loss * M_prime, "G": Lasso, "frame_expansion": frame_expansion_config}


def fe_huber_non_neg_reg(Loss, M, data_shape, lambda_, **kwargs):
    print(f"Executing huber non negative, lambda={lambda_}")
    Derivative = FirstDerivative(size=content_size, kind="forward")
    Derivative.compute_lipschitz_cst(tol=1e-2)

    # This functional is differentiable
    F_prime = lambda_ * HuberNorm(M=content_size, delta=0.001) * Derivative
    # hence can be optimized using gradient descent
    F = Loss * (M * FrameExpansion(dim=data_shape, margins=content_margins)) + F_prime

    G = NonNegativeOrthant(dim=content_size)

    return {"F": F, "G": G, "frame_expansion": frame_expansion_config}


##############################
#   Export regularizations   #
##############################

"""
Available Regularization strategies
A Regularization Strategy is made of two parts:
`Regularization`: `func`
    Function receiving a loss functional (of type `Callable`) and returning
    the functionals to optimize (`F`, `G` or `H`)
`Optimizer`: `func`
    Function running the appropriate optimizer and returning the reconstructed signal,
    together with additional information on the execution
`EstimateRecovery`: `func`
    Function converting the result of the optimization problem into an estimate of the image
"""
available_regs: dict[str:(Regularization, Optimizer, EstimateRecovery)] = {
    # WITHOUT frame expansion
    "l2": (tikhonov_reg, apgd_optim, trivial_estimate_recovery),
    "lasso": (lasso_reg, apgd_optim, trivial_estimate_recovery),
    "non-neg": (non_negative_reg, apgd_optim, trivial_estimate_recovery),
    "dct": (dct_lasso_reg, apgd_optim, dct_estimate_recovery),
    "tv-non-neg": (tv_non_negative_reg, pds_optim, trivial_estimate_recovery),
    "huber-non-neg": (huber_non_neg_reg, apgd_optim, trivial_estimate_recovery),
    # WITH frame expansion
    "fe-lasso": (fe_lasso_reg, apgd_optim, trivial_estimate_recovery),
    # "fe-dct":       (fe_dct_lasso_reg,      apgd_optim,  dct_estimate_recovery),
    "fe-huber": (fe_huber_non_neg_reg, apgd_optim, trivial_estimate_recovery),
}
