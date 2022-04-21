**This is repository of thesis made by Sharifullina Ramilya.
Skolkovo institute of science and technology.
Petroleum engineering programme (2022).**


The latest developments are in the folder `TIME`.

The `TIME` folder consists of the following files:
1. The `dataset.py` contains 
2. The `model.py` contains original GAN model.
There are modeifications of this model: 
	- `model_one_side_labeling` 
	- ...
3. `time_series_metrics` contains of
	- `ssim_loss.py` where loss function for 1d structure similarity is written
	- `tsfresh_metrics.py` where metrics for time series comparison are written, loss function for such metrics will be written further
4. `model_verification.pt` is saved verification (inception time) model paramteres.

5. `train.ipynb` where all computations are made 
