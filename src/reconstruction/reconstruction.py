import time
from multiprocessing import Pool
from typing import Optional

import numpy as np

from src.reconstruction.pipelines import (
    compute_loss,
    optimize,
    setup_loss,
    setup_regularization,
)

# Custom zoom for our dataset

# FRAME 1
# zoom_shape = (270, 480)
# zoom_portion = ((150, 420), (260, 740))
# ZOOM 2
# zoom_shape = (261, 464)
# zoom_portion = ((155, 416), (268, 732))
# ZOOM 3 (Perfectly centered)
zoom_shape = (243, 432)
zoom_portion = ((172, 415), (278, 710))


def zoom_over_content(img):
    if img.shape == zoom_shape:
        return img
    else:
        return img[zoom_portion[0][0] : zoom_portion[0][1], zoom_portion[1][0] : zoom_portion[1][1]]


def reconstruct(
    data: np.ndarray,
    psf: np.ndarray,
    state: dict,
    lmb: float,
    reg: str,
    gray: Optional[bool] = False,
    parallel: Optional[bool] = True,
    index: Optional[int] = 0,
):
    final_loss = None
    rec_img = None

    if not gray:
        print("Processing 3 color channels")

        channels = [
            {
                "state": state,
                "data": data[:, :, ch],
                "psf": psf[:, :, ch],
                "lmb": lmb,
                "reg": reg,
                "test_num": index,
                "channel_name": as_color_name(ch),
            }
            for ch in range(3)
        ]

        reconstruction_res = spawn_reconstruction(channels, parallel)

        rec_img_s = []
        losses = []
        for (ch_image, ch_loss) in reconstruction_res:
            rec_img_s.append(zoom_over_content(ch_image))
            losses.append(ch_loss)

        # Mean of losses
        final_loss = np.array(losses, dtype=float).mean()

        rec_img = np.stack(list(rec_img_s), axis=-1)

    else:
        rec_img, final_loss = spawn_reconstruction(
            [
                {
                    "state": state,
                    "data": data,
                    "psf": psf,
                    "lmb": lmb,
                    "reg": reg,
                    "test_num": index,
                    "channel_name": "GRAY",
                }
            ],
            parallel,
        )[0]
        rec_img = zoom_over_content(rec_img)

    print(f"Final_loss={final_loss:.10f}")

    # 'image_name', 'regularisation', 'lambda', 'mse', 'psnr', 'ssim', 'lpips', 'final_loss', 'timestamp'
    return final_loss, rec_img.astype(np.float32)


def spawn_reconstruction(channels, is_parallel):
    """
    Runs several reconstruction process, possibly exploing parallelization
    """
    if is_parallel:
        print("[INFO] Launching parallel executions")
        with Pool(processes=3) as pool:
            res = pool.map(_serial_reconstruction, channels)

    else:
        print("[WARNING] Parallelization is disabled. Reverting to serial execution")
        res = []
        for channel in channels:
            res.append(_serial_reconstruction(channel))

    return res


def _serial_reconstruction(channel):
    state = dict(channel["state"])
    data = channel["data"]
    psf = channel["psf"]
    lmb = channel["lmb"]
    reg = channel["reg"]
    test_num = channel["test_num"]
    channel_name = channel["channel_name"]

    ### Reconstruction setup ###
    start_time = time.time()

    state.update(setup_loss(data, psf))

    # Apply test configuration
    test_config = {
        "lambda_": lmb,
    }
    state.update(test_config)

    state = setup_regularization(reg, state)
    print(
        f"[Instance #{test_num}/{channel_name}] :: objective setup time {time.time() - start_time:.3f} s"
    )

    ### Reconstruction ###
    start_time = time.time()

    state = optimize(reg, state)
    print(
        f"[Instance #{test_num}/{channel_name}] :: reconstruction time {time.time() - start_time:.3f} s"
    )
    print(
        f"[Instance #{test_num}/{channel_name}] :: Convergence={state['has_converged']} \t Iterations={state['iters_num']}"
    )

    final_loss = compute_loss(state)

    # Detect use of frame expansion
    if "frame_expansion" in state:
        reconstructed_shape = state["frame_expansion"]["content_shape"]
    else:
        reconstructed_shape = state["data_shape"]

    img_data = state["reconstructed"].reshape(reconstructed_shape)

    # normalize
    img_data = np.clip(img_data, a_min=0, a_max=data.max())
    img_data /= np.linalg.norm(img_data.ravel())

    return img_data, final_loss


def as_color_name(channel_id):
    if channel_id == 0:
        return "RED  "
    elif channel_id == 1:
        return "GREEN"
    else:
        return "BLUE "
