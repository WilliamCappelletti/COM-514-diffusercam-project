"""Script to optimize hyperparameters for every image defined in src.data.config"""
from warnings import simplefilter

simplefilter("ignore", UserWarning)

import click
import matplotlib.pyplot as plt
from diffcam import plot
from diffcam.io import load_data
from diffcam.util import resize

import src.data.config as cfg
from src.data.io import init_results
from src.reconstruction.hyperopt import hyperoptimize
from src.reconstruction.reconstruction import reconstruct, zoom_shape


@click.command()
def main():
    """Script to optimize hyperparameters for every image defined in src.data.config"""
    # Load/init results
    results_df = init_results(cfg.PATHS.results_df)
    cfg.PATHS.results_df.parent.mkdir(parents=True, exist_ok=True)

    # Test
    for img_idx, reg_list in cfg.TO_TEST.items():
        if not reg_list:
            continue

        img_name = cfg.FILE.fmt_proc.format(img_idx)
        raw_name = cfg.FILE.fmt_raw.format(img_idx)

        try:
            # Load images
            psf, data, ground_truth_img = load_data(
                psf_fp=str(cfg.PATHS.psf),
                data_fp=str(cfg.PATHS.processed / img_name),
                data_truth_fp=str(cfg.PATHS.raw / raw_name),
                downsample=cfg.FILE.downsample,
            )
        except Exception as err:
            with open("errors.log", "a", encoding="utf-8") as f:
                f.write(f"Error {err.__class__.__name__} in file {img_name}")
            continue

        ground_truth_img = resize(ground_truth_img, zoom_shape[0] / ground_truth_img.shape[0])

        state = {
            "data_size": data.size // 3,
            "data_shape": (data.shape[0], data.shape[1]),
            "min_iter": 0,
            "max_iter": 1000,
        }

        for reg in reg_list:
            print(f"[INFO] Tuning on {img_name} with {reg}")
            # Call hyperoptimize
            try:
                best_lambda, best_val = hyperoptimize(
                    data,
                    psf,
                    ground_truth_img,
                    state,
                    reg,
                    objective=cfg.HPSearch.objective,
                    n_trials=cfg.HPSearch.nb_trials,
                )
            except Exception as err:
                with open("errors.log", "a", encoding="utf-8") as f:
                    f.write(f"Error {err.__class__.__name__} in reg {reg} for file {img_name}\n")
                continue

            # Add results to dataframe
            results_df = results_df.append(
                {
                    "img_name": img_name,
                    "objective": cfg.HPSearch.objective,
                    "reg": reg,
                    "reg_lambda": best_lambda,
                    "reg_val": best_val,
                    "nb_trials": cfg.HPSearch.nb_trials,
                },
                ignore_index=True,
            )

            # Save dataframe
            results_df.to_csv(cfg.PATHS.results_df)

            # Reconstruct and save image
            *_, rec_image = reconstruct(data, psf, state, best_lambda, reg, index=-1)

            plot.plot_image(rec_image)
            plt.savefig(cfg.PATHS.results / cfg.FILE.fmt_rec.format(img_idx, reg))
            plt.close()


if __name__ == "__main__":
    main()
