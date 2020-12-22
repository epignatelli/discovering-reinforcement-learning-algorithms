![Build](https://github.com/epignatelli/discovering-reinforcement-learning-algorithms/workflows/build/badge.svg)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Discovering reinforcement learning algorithms
A jax/stax implementation of the NeurIPS 2020 paper: _Discovering reinforcement learning algorithms_ [[1]](https://proceedings.neurips.cc/paper/2020/file/0b96d81f0494fde5428c7aea243c9157-Paper.pdf)

The agent at `lpg.agent.py` implements the `bsuite.baseline.base.Agent` interface.
The `lpg/environments/*.py` interfaces with a `dm_env.Environment`.
We wrap the [gym-atari](https://github.com/openai/gym) suite using the `bsuite.utils.gym_wrapper.DMEnvFromGym` adapter into a `dqn.AtariEnv` to implement historical observations and actions repeat.


## Installation
To run the algorithm on a GPU, I suggest to [install](https://github.com/google/jax#pip-installation) the gpu version of `jax` [[4]](https://github.com/google/jax). You can then install this repo using [Anaconda python](https://www.anaconda.com/products/individual) and [pip](https://pip.pypa.io/en/stable/installing/).
```sh
conda env create -n lpg
conda activate lpg
pip install git+https://github.com/epignatelli/discovering-reinforcement-learning-algorithms
```


## References
[1] [_Oh, J., Hessel, M., Czarnecki, W.M., Xu, Z., van Hasselt, H.P., Singh, S. and Silver, D., 2020. Discovering reinforcement learning algorithms. Advances in Neural Information Processing Systems, 33._](https://proceedings.neurips.cc/paper/2020/file/0b96d81f0494fde5428c7aea243c9157-Paper.pdf)
