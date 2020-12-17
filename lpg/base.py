import functools
from typing import Any, Callable, NamedTuple, Tuple
import jax.numpy as jnp

RNGKey = jnp.ndarray
Shape = Tuple[int, ...]
Params = Any


def factory(cls_maker, T):
    @functools.wraps(cls_maker)
    def fabricate(*args, **kwargs):
        return T(*cls_maker(*args, **kwargs))

    return fabricate


class Module(NamedTuple):
    init: Callable[[RNGKey, Shape], Tuple[Shape, Params]]
    apply: Callable[[Params, jnp.ndarray], jnp.ndarray]


class RnnCell(NamedTuple):
    initial_state: Callable[[], jnp.ndarray]
    init: Callable[[RNGKey, Shape], Tuple[Shape, Params]]
    apply: Callable[[Params, jnp.ndarray], jnp.ndarray]


module = functools.partial(factory, T=Module)
rnn_cell = functools.partial(factory, T=RnnCell)