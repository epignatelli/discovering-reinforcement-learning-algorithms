import jax
import jax.numpy as jnp
from lpg.agent import HParams, Lpg


def test_lpg():
    hparams = HParams(hidden_size=32)
    meta = Lpg(hparams)

    m = 3
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    x = (r, d, gamma, pi, y, y_t) = (
        jnp.array([1.0]),
        jnp.array([0.0]),
        jnp.array([0.95]),
        jnp.array([0.7]),
        jax.random.normal(k1, (m,)),
        jax.random.normal(k2, (m,)),
    )

    output_shape, params = meta.init(rng, ((1,), (1,), (1,), (1,), (m,), (m,)))
    y_hat, pi = meta.apply(params, x)
    # check that outputs are scalar
    float(y_hat)
    float(pi)
    print("y_hat:", y_hat, "pi", pi)


test_lpg()