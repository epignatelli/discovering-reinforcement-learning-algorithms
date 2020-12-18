import jax
import jax.numpy as jnp
from lpg.agent import HParams, Lpg


def test_lpg():
    hparams = HParams(hidden_size=32)
    meta = Lpg(hparams)

    rng = jax.random.PRNGKey(0)
    input_shape = (16,)
    x = jax.random.normal(rng, input_shape)
    output_shape, params = meta.init(rng, input_shape)
    y_hat, pi = meta.apply(params, x)
    print("y_hat:", y_hat, "pi", pi)


test_lpg()