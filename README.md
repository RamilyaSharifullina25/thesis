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

4. `inceptiontime` is inception time classifier model

5. `trained_models` is a folder where all pretrained models are located including:  
	- `model_verification.pt` is saved verification (inception time) model paramteres.  
	- `gan_ms`  
	- `gan_rms`  
	- `gan_cba`  
	- ...  

6. `researchs_256window.ipynb` and `researchs_64window.ipynb` where all computations are made 

7. `models_new_64.py` and `models_new_256.py` are notebooks where Generator and Discrimiator are contained  
