from typing import Dict

import numpy as np
from diffcam.metric import lpips, mse, psnr, ssim


def score_image(rec_img: np.ndarray, ground_truth_img: np.ndarray) -> Dict[str, float]:
    mse_res = mse(ground_truth_img, rec_img)
    psnr_res = psnr(ground_truth_img, rec_img)
    ssim_res = ssim(ground_truth_img, rec_img)
    lpips_res = lpips(ground_truth_img, rec_img)

    return {"mse": mse_res, "psnr": psnr_res, "ssim": ssim_res, "lpips": lpips_res}
