#!/bin/bash
"""
```bash

python src/models/reconstruction.py \
--psf_fp DiffuserCam/data/psf/diffcam_rgb.png \
--data_fp DiffuserCam/data/raw_data/thumbs_up_rgb.png \
--data_truth_fp DiffuserCam/data/raw_data/thumbs_up_rgb.png \
--h_
--reg lasso --save --parallel --preview
```
"""
import os
import pathlib as plib
import time
from datetime import datetime

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diffcam import plot
from diffcam.io import load_data
from diffcam.util import resize

from src.reconstruction.hyperopt import hyperoptimize
from src.reconstruction.pipelines import available_regs
from src.reconstruction.reconstruction import reconstruct, zoom_shape
from src.reconstruction.score import score_image


@click.command()
@click.option(
    "--psf_fp",
    type=click.Path(exists=True),
    help="File name for recorded PSF.",
)
@click.option(
    "--data_fp",
    type=click.Path(exists=True),
    help="File name for raw measurement data.",
)
@click.option(
    "--data_truth_fp",
    type=click.Path(exists=True),
    help="File name for ground truth image",
)
@click.option(
    "--n_iter",
    default=30,
    type=int,
    help="Number of iterations.",
)
@click.option(
    "--reg_lambda",
    type=float,
    default=None,
    help="Regularizer lambda",
)
@click.option(
    "--hp_objective",
    default=None,
    required=False,
    type=click.Choice(["mse", "psnr", "ssim", "lpips"]),
    help="Hyperparameter tuning objective",
)
@click.option(
    "--n_hp_trials",
    default=50,
    type=int,
    help="Number of hyperparameter optimization trials.",
)
@click.option(
    "--downsample",
    type=float,
    default=4,
    help="Downsampling factor.",
)
@click.option(
    "--disp",
    default=50,
    type=int,
    help="How many iterations to wait for intermediate plot/results. Set to negative value for no intermediate plots.",
)
@click.option(
    "--flip",
    is_flag=True,
    help="Whether to flip image.",
)
@click.option("--preview", is_flag=True, help="Whether to preview the image after reconstruction")
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save intermediate and final reconstructions.",
)
@click.option(
    "--save_dir",
    type=click.Path(exists=True, dir_okay=True),
    default="./data/results",
    help="Relative/Absolute path to the directory in which output files are saved (MUST end with a slash)",
)
@click.option(
    "--gray",
    is_flag=True,
    help="Whether to perform construction with grayscale.",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
)
@click.option(
    "--no_plot",
    is_flag=True,
    help="Whether to no plot.",
)
@click.option(
    "--bg",
    type=float,
    help="Blue gain.",
)
@click.option(
    "--rg",
    type=float,
    help="Red gain.",
)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--reg",
    default="l1",
    type=click.Choice(available_regs.keys()),
    help="Regularization function",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)
@click.option("--parallel", is_flag=True, help="Enable parallelization of image reconstruction")
def reconstruction(
    psf_fp,
    data_fp,
    data_truth_fp,
    n_iter,
    reg_lambda,
    hp_objective,
    n_hp_trials,
    downsample,
    disp,
    flip,
    gray,
    bayer,
    bg,
    rg,
    gamma,
    preview,
    save,
    no_plot,
    reg,
    single_psf,
    parallel,
    save_dir,
):
    """
    Load data and reconstruct image with given parameters. If `hp_objective` is set, the hyperparameter
    `lambda` is optimized automatically. If `parallel` is set, the function exploits multicore processing
    (1 core for color channel)

    Parameters:

    --psf_fp PATH                   File name for recorded PSF.
    --data_fp PATH                  File name for raw measurement data.
    --data_truth_fp PATH            File name for ground truth image
    --n_iter INTEGER                Number of iterations.
    --reg_lambda FLOAT              Regularizer lambda
    --hp_objective [mse|psnr|ssim|lpips]
                                    Hyperparameter tuning objective
    --n_hp_trials INTEGER           Number of hyperparameter optimization
                                    trials.
    --downsample FLOAT              Downsampling factor.
    --disp INTEGER                  How many iterations to wait for intermediate
                                    plot/results. Set to negative value for no
                                    intermediate plots.
    --flip                          Whether to flip image.
    --preview                       Whether to preview the image after
                                    reconstruction
    --save                          Whether to save intermediate and final
                                    reconstructions.
    --save_dir PATH                 Relative/Absolute path to the directory in
                                    which output files are saved (MUST end with
                                    a slash)
    --gray                          Whether to perform construction with
                                    grayscale.
    --bayer                         Whether image is raw bayer data.
    --no_plot                       Whether to no plot.
    --bg FLOAT                      Blue gain.
    --rg FLOAT                      Red gain.
    --gamma FLOAT                   Gamma factor for plotting.
    --reg [l2|lasso|non-neg|dct|tv-non-neg|huber-non-neg|fe-lasso|fe-huber]
                                    Regularization function
    --single_psf                    Same PSF for all channels (sum) or unique
                                    PSF for RGB.
    --parallel                      Enable parallelization of image
                                    reconstruction
    --help                          Show this message and exit.
    """

    psf, data, ground_truth_img = load_data(
        psf_fp=psf_fp,
        data_fp=data_fp,
        data_truth_fp=data_truth_fp,
        downsample=downsample,
        bayer=bayer,
        blue_gain=bg,
        red_gain=rg,
        plot=not no_plot,
        flip=flip,
        gamma=gamma,
        gray=gray,
        single_psf=single_psf,
    )

    if disp < 0:
        disp = None

    if save:
        save = os.path.basename(data_fp).split(".")[0]
        save = save_dir + save
        print(save)

    ### Warmup ###
    start_time = time.time()

    shape = data.shape

    if n_iter:
        (min_iter, max_iter) = (1, n_iter)
    else:
        (min_iter, max_iter) = (100, 200)

    if gray:
        state = {
            "data_size": data.size,
            "data_shape": shape,
            "min_iter": min_iter,
            "max_iter": max_iter,
        }
    else:
        state = {
            "data_size": data.size // 3,
            "data_shape": (shape[0], shape[1]),
            "min_iter": min_iter,
            "max_iter": max_iter,
        }

    if single_psf and not gray:
        # Must replicate psf over RGB channels
        psf = np.repeat(psf, 3, axis=2)

    if data_truth_fp:
        # Resize true image
        ground_truth_img = resize(ground_truth_img, zoom_shape[0] / ground_truth_img.shape[0])

    cols = [
        "image_name",
        "regularisation",
        "lambda",
        "mse",
        "psnr",
        "ssim",
        "lpips",
        "final_loss",
        "timestamp",
        "runtime",
    ]
    df_res = pd.DataFrame(columns=cols)
    image_name = os.path.basename(data_fp)

    print(f"Warmup time : {time.time() - start_time:.3f} s")

    if reg_lambda is not None:
        lmb = reg_lambda
    elif hp_objective is not None:
        if ground_truth_img is None:
            raise ValueError("Need ground truth image to perform hyper-optimization")
        print("[INFO] Starting hyper-optimization")
        lmb, *_ = hyperoptimize(
            data,
            psf,
            ground_truth_img,
            state,
            reg,
            objective=hp_objective,
            n_trials=n_hp_trials,
        )
        # print(f"[INFO] Best lambda found out of {n_hp_trials} is {lmb:.3e}")
    else:
        raise ValueError(
            "Need either hp_objective or reg_lambda (specify 'reg_lambda' for direct reconstruction or 'hp_objective' for lambda tuning)"
        )

    start_time = time.time()

    final_loss, rec_img = reconstruct(data, psf, state, lmb, reg, gray, parallel, 0)

    runtime = time.time() - start_time

    if ground_truth_img is not None:

        ground_truth_img = ground_truth_img.astype(np.float32)
        mse_res, psnr_res, ssim_res, lpips_res = score_image(rec_img, ground_truth_img).values()

        print(
            f"[INFO] :: Final MSE={mse_res} \t psnr={psnr_res} \t ssim={ssim_res} \t lpips={lpips_res}"
        )

    else:
        mse_res = psnr_res = ssim_res = lpips_res = -1

    timestamp = datetime.now().strftime("%d%m%H%M%S")

    # 'image_name', 'regularisation', 'lambda', 'mse', 'psnr', 'ssim', 'lpips', 'final_loss', 'timestamp'
    df_res.loc[0] = [
        image_name,
        reg,
        lmb,
        mse_res,
        psnr_res,
        ssim_res,
        lpips_res,
        final_loss,
        timestamp,
        runtime,
    ]

    if preview:
        plot.plot_image(rec_img)
        plt.title(reg + f" (lambda={lmb:.0e})")
        plt.show()

    if save:
        file_name = (
            f"{save_dir}/rec_{plib.Path(image_name).stem}_{str(reg)}"
            f"_{str(n_iter) if n_iter else 'auto'}_{lmb:.0e}_{timestamp}.png"
        )

        print("[INFO] Saving to", file_name)

        plot.plot_image(rec_img)
        plt.savefig(file_name)

    if save:
        # Pick first timestamp in DataFrame
        timestamp = str(timestamp)
        save_df = save_dir + "/" + "log_" + timestamp + ".csv"
        print(df_res)
        df_res.to_csv(save_df)


def crop(img, x_up, x_bottom, y_up, y_bottom):
    x, y = img.shape
    return img[x_up : x - x_bottom, y_up : y - y_bottom]


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    reconstruction()
