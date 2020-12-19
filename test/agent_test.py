import jax
import jax.numpy as jnp
from lpg.agent import HParams, Lpg


def test_lpg():
    hparams = HParams(hidden_size=32)
    agent = Lpg(hparams)


test_lpg()