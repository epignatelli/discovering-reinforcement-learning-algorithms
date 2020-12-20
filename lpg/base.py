import functools
from typing import Any, Callable, NamedTuple, Tuple
import jax
import jax.numpy as jnp

RNGKey = jnp.ndarray
Shape = Tuple[int, ...]
Params = Any


def inject(cls, *args, **kwargs):
    def decorate(fun):
        # TODO(epignatelli): remove cls argument, infer it, instead
        setattr(
            cls,
            fun.__name__,
            jax.jit(fun, *args, **kwargs),
        )

        def wrapper(*a, **k):
            return fun(*a, **k)

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
