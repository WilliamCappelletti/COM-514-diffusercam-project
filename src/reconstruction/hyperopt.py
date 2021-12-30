from typing import Tuple

import numpy as np
import optuna

from src.reconstruction.reconstruction import reconstruct
from src.reconstruction.score import score_image


def hyperoptimize(
    data: np.ndarray,
    psf: np.ndarray,
    ground_truth_img: np.ndarray,
    state: dict,
    reg: str,
    objective: str,
    n_trials: int,
) -> Tuple[float, float]:
    # Speedup
    state = dict(state)
    state["max_iter"] = 600

    def objective_func(trial: optuna.trial.Trial):
        lmb = trial.suggest_float("reg_lambda", low=1e-8, high=1e-1)
        _, rec_image = reconstruct(data, psf, state, lmb, reg, index=trial.number)
        score_dict = score_image(rec_img=rec_image, ground_truth_img=ground_truth_img)

        print(
            f"[Iteration #{trial.number}] :: MSE={score_dict['mse']} \t psnr={score_dict['psnr']} "
            f"\t ssim={score_dict['ssim']} \t lpips={score_dict['lpips']}"
        )

        return score_dict[objective]

    opti_direction = {
        "mse": "minimize",
        "psnr": "maximize",
        "ssim": "maximize",
        "lpips": "minimize",
    }

    study = optuna.create_study(direction=opti_direction[objective])
    study.optimize(objective_func, n_trials=n_trials)

    return study.best_params["reg_lambda"], study.best_trial.value
