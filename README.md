# Density Ratio Estimation via Infinitesimal Classification

This repo contains a reference implementation for DRE-<img src="https://render.githubusercontent.com/render/math?math=\infty"> as described in the paper:
> Density Ratio Estimation via Infinitesimal Classification </br>
> [Kristy Choi*](http://kristychoi.com/), [Chenlin Meng*](https://cs.stanford.edu/~chenlin/), [Yang Song](https://yang-song.github.io/), [Stefano Ermon](https://cs.stanford.edu/~ermon/) </br>
> International Conference on Artificial Intelligence and Statistics (AISTATS), 2022. [ORAL] </br>
> Paper: https://arxiv.org/abs/2111.11010 </br>

Note that the code structure is a direct extension of: https://github.com/yang-song/score_sde_pytorch


## Environment setup:
(a) Necessary packages can be found in `requirements.txt`.

(b) Set the correct Python path using the following command:
```
source init_env.sh
```
(c) Note that we use `wandb` for keeping track of train/test statistics. You will have to set up wandb in the `main.py` file.

## For the 1-D peaked Gaussian experiments:
To train a time score network with the time-score matching loss *only*, run:
```
python3 main.py --toy --config configs/1d_gaussians/time/param.py --mode=train \
--doc=1d_peaked_gaussians_param_time \
--workdir=./results/1d_peaked_gaussians_param_time/
 ```

 To train a joint score network with both the time and data scores, run: 
 ```
 python3 main.py --toy --config configs/1d_gaussians/joint/param.py --mode=train \
--doc=1d_peaked_gaussians_param_joint \
--workdir=./results/1d_peaked_gaussians_param_joint/
```

## For the MI estimation experiments using the joint score matching objective:
(40-D)
```
python3 main.py --toy  \
  --config configs/gmm_mutual_info/joint/param.py --mode=train  \
  --doc=mi_40d_param_joint --config.model.type=joint \
  --config.training.joint=True --config.data.dim=40  \
  --config.seed=7777 --config.training.n_iters=20001  \
  --workdir=./results/gmm_mi_40d_param_joint/  \
  --config.training.batch_size=512 --config.training.eval_freq=2000 \
  --config.training.reweight=True
```
For 80-D, we set `config.data.dim=80`, `config.training.n_iters=50001`, `config.training.eval_freq=5000`. 

For 160-D, we set `config.data.dim=160`, `config.training.eval_freq=5000`, and `config.training.n_iters=200001`. 

For 320-D, we set `config.data.dim=320`, `config.training.eval_freq=8000`, `config.training.batch_size=256`, and `config.training.n_iters=400001`.


## For the MNIST experiments:
First, we use the `nsf` codebase to train the flow models. All pre-trained model checkpoints (Gaussian, Copula, RQ-NSF) can be found in `flow_ckpts/`. There is no need to re-train the flow models from scratch and all the time score networks take into account the particular ways that the data has been preprocessed.

(a) For the Gaussian noise model:
```
python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_noise \
--workdir=./results/mnist_z_unet_lin_emb_noise
```

(b) For the copula:
```
python3 main.py --flow \
--config configs/mnist/z_copula_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_copula \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_copula
```

(c) For the RQ-NSF flow model:
```
python3 main.py --flow \
--config configs/mnist/z_flow_time_interpolate.py \
--mode=train --doc=z_unet_lin_emb_flow \
--config.training.likelihood_weighting=True \
--workdir=./results/mnist_z_unet_lin_emb_flow
```

## PENDING: evaluation code for AIS


## References
If you find this work useful in your research, please consider citing the following paper:
```
@article{choi2021density,
  title={Density Ratio Estimation via Infinitesimal Classification},
  author={Choi, Kristy and Meng, Chenlin and Song, Yang and Ermon, Stefano},
  journal={arXiv preprint arXiv:2111.11010},
  year={2021}
}
```
