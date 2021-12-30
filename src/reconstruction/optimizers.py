from pycsou.opt.proxalgs import APGD, PrimalDualSplitting


def apgd_optim(F, data_size, min_iter, max_iter, G=None, frame_expansion=None, **kwargs):
    """
    Plain APGD optimizer
    """
    # Detect use of frame expansion
    if frame_expansion:
        problem_size = frame_expansion["content_size"]
    else:
        problem_size = data_size

    apgd = APGD(dim=problem_size, min_iter=min_iter, max_iter=max_iter, F=F, G=G, verbose=None)
    estimate, converged, diagnostics = apgd.iterate()

    iters_num = diagnostics.shape[0] - 1

    return {
        "has_converged": converged,
        "iters_num": iters_num,
        "reconstructed": estimate["iterand"],
    }


def pds_optim(F, G, H, K, data_size, min_iter, max_iter, **kwargs):
    """
    Plain PDS optimizer
    """
    pds = PrimalDualSplitting(
        dim=data_size, min_iter=min_iter, max_iter=max_iter, F=F, G=G, H=H, K=K, verbose=None
    )

    estimate, converged, diagnostics = pds.iterate()

    iters_num = diagnostics.shape[0] - 1

    return {
        "has_converged": converged,
        "iters_num": iters_num,
        "reconstructed": estimate["primal_variable"],
    }
