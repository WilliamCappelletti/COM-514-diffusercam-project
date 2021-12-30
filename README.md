DiffuserCam
==============================

DiffuserCam source code

Setup
-----

Init the repository and create a conda environment with required packages by running
```sh
make env
```
You can activate the environment with
```sh
conda activate diffusercam
```

If the requirements change, you can update the environment using
```sh
make requirements
```


Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make env` or `make requirements`
    ├── data
    │   ├── interim        <- Directory containing our PSF function
    │   ├── processed      <- Color-corrected photos taken by the DiffuserCam
    │   ├── raw            <- The original photos
    │   └── reconstructed  <- Reconstructed images
    │
    ├── notebooks          
    │   └── convolve_benchmark.ipynb
    |                      <- Contains the performance tests for Convolve2D_fft
    |
    ├── setup.py          
    │
    ├── reports            
    │   └── report.pdf     <- Our report in .pdf   
    │
    ├── requirements.yml   <- The requirements file for reproducing the analysis environment with conda
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── reconstruction
    │   │   ├── convolution_fft.py      <- Optimized 2D convolution operator
    │   │   ├── dct.py                  <- Discrete Cosine Transform operator
    │   │   ├── frame_expansion.py      <- Frame Expansion operator (see report for details)
    │   │   ├── hubernorm.py            <- Huber Norm operator
    │   │   ├── hyperopt.py             <- Optimization functions
    │   │   ├── main.py                 <- Entry-point for image reconstruction
    │   │   ├── optimizers.py           <- Wrappers of Pycsou optimization methods
    │   │   ├── pipelines.py            <- Code to handle the different stages of reconstruction
    │   │   ├── reconstruction.py       <- Code for reconstruction parallelization
    │   │   ├── regularizations.py      <- Code for regularization strategies (regularization + optimizer + ...)
    │   │   └── score.py                <- Code for computing metrics on images
    |   |
    │   ├── data                        <- Code for finetuning parameters on our dataset
    │   │   ├── config.py               <- Reconstruction parameters
    │   │   ├── io.py                   <- Code for loading the configuration
    │   │   └── tuning.py               <- Entry-point

--------

Useful commands
------------

To reconstruct an image, use `src/reconstruction/main.py` with the following flags

```
Options:
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
```

### Examples

#### Single image reconstruction

```
python ./src/reconstruction/main.py --psf_fp "./data/interim/psf_rgb.png" --data_fp "./data/processed/photo10_rgb.png" --reg lasso --n_iter 600 --reg_lambda 1.5e-7 --parallel --preview --save
```

reconstructs the 10th image of the dataset, using LASSO regularization and saving the result in `./data/results/`

#### Hyperparameter tuning over all the images in the dataset
```
python ./src/data/tuning.py
```

loads the configuration in `./src/data/config.py` and finds, for each image and regularization strategy, the best hyperparameter; saves the logs in `./data/results/hp.csv`.
