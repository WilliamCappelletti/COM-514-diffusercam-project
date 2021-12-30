"""Parameters for optimization of regularizers lambda for reconstruction of our dataset"""
from pathlib import Path


class FILE:
    fmt_proc = "photo{}_rgb.png"
    fmt_raw = "c_photo{}.jpeg"
    fmt_rec = "photo{}_rec_{}.png"

    downsample = 4


class PATHS:
    _root = Path("data")
    raw = _root / "raw"
    captures = _root / "captures"
    processed = _root / "processed"
    psf = _root / "interim/psf_rgb.png"
    results = _root / "results"
    results_df = results / "hp.csv"


class HPSearch:
    nb_trials = 20
    objective = "lpips"


# Dict of images to regularizations meaningful to test
# Possible regs:
#   - l2: smoothness
#   - lasso: sparse
#   - non-neg: reasonable for imgs
#   - dct: sparse in freq domain
#   - tv-non-neg: piecewise const
#   - huber-non-neg: same as TV but differentiable
#   - fe-lasso: frame expasion version of lasso
#   - fe-huber: frame expasion version of huber

TO_TEST = {
    "01": ["lasso", "non-neg", "l2", "dct"],
    "02": [],
    "03": [],
    "04": [],
    "05": [],
    "06": ["dct", "non-neg", "lasso", "l2"],
    "07": ["dct", "non-neg", "l2"],
    "08": ["huber-non-neg", "lasso", "non-neg"],
    "09": ["huber-non-neg", "lasso", "dct"],
    "10": ["huber-non-neg", "lasso", "dct", "non-neg"],
    "11": ["huber-non-neg", "l2", "dct"],
    "12": ["huber-non-neg", "l2", "non-neg"],
    "13": ["dct", "huber-non-neg", "lasso"],
    "14": ["dct", "huber-non-neg"],
    "15": ["huber-non-neg", "l2", "dct"],
    "16": ["huber-non-neg", "lasso", "dct", "non-neg"],
}
