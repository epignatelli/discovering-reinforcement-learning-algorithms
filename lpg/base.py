import functools
from typing import Any, Callable, NamedTuple, Tuple
import jax
import jax.numpy as jnp

RNGKey = jnp.ndarray
Shape = Tuple[int, ...]
Params = Any


def inject(cls, static_argnums=None):
    def decorate(function):
        # TODO(epignatelli): remove cls argument, infer it, instead
        setattr(
            cls,
            function.__name__,
            function
            if not static_argnums
            else jax.jit(function, static_argnums=static_argnums),
        )

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorate


def factory(cls_maker, T):
    @functools.wraps(cls_maker)
    def fabricate(*args, **kwargs):
        return T(*cls_maker(*args, **kwargs))

    return fabricate


class Module(NamedTuple):
    init: Callable[[RNGKey, Shape], Tuple[Shape, Params]]
    apply: Callable[[Params, jnp.ndarray], jnp.ndarray]
    initial_state: Callable[[], jnp.ndarray] = None


module = functools.partial(factory, T=Module)
