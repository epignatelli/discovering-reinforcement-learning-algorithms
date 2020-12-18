import functools
from typing import Any, Callable, NamedTuple, Tuple
import jax.numpy as jnp

RNGKey = jnp.ndarray
Shape = Tuple[int, ...]
Params = Any
Interface = NamedTuple


def factory(cls_maker, T):
    @functools.wraps(cls_maker)
    def fabricate(*args, **kwargs):
        return T(*cls_maker(*args, **kwargs))

    return fabricate


class Module(Interface):
    init: Callable[[RNGKey, Shape], Tuple[Shape, Params]]
    apply: Callable[[Params, jnp.ndarray], jnp.ndarray]
    initial_state: Callable[[], jnp.ndarray] = None


module = functools.partial(factory, T=Module)