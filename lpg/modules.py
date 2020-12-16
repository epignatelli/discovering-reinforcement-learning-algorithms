import functools
from typing import Any, Callable, NamedTuple, Tuple
import jax.numpy as jnp
import jax
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal, zeros
from jax.experimental.stax import Dense

RNGKey = jnp.ndarray
Shape = Tuple[int, ...]
Params = Any
CellState = Any


def factory(cls_maker, T):
    @functools.wraps(cls_maker)
    def fabricate(*args, **kwargs):
        out = cls_maker(*args, **kwargs)
        return T(*out)

    return fabricate


class Module(NamedTuple):
    init: Callable[[RNGKey, Shape], Tuple[Shape, Params]]
    apply: Callable[[Params, jnp.ndarray], jnp.ndarray]


module = functools.partial(factory, T=Module)


@module
def Rnn(cell: Module, n_layers: int):
    def init(rng, input_shape):
        return cell.init(rng, input_shape)

    def apply(params, inputs, prev_state=cell.initial_state()):
        outputs, state = jax.lax.fori_loop(
            0,
            n_layers,
            lambda i, val: cell.apply(params, val[0], val[1]),
            (inputs, prev_state),
        )
        return outputs

    return (init, apply)


class LSTMState(NamedTuple):
    h: jnp.ndarray
    c: jnp.ndarray


@module
def LSTMCell(
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    h_initial_state_fn=zeros,
    c_initial_state_fn=zeros,
    initial_state_seed=0,
):
    """Layer construction function for an LSTM cell.
    Formulation: Zaremba, W., 2014, https://arxiv.org/pdf/1409.2329.pdf"""

    def initial_state():
        shape = (hidden_size,)
        k1, k2 = jax.random.split(jax.random.PRNGKey(initial_state_seed))
        return LSTMState(h_initial_state_fn(k1, shape), c_initial_state_fn(k2, shape))

    def init(rng, input_shape):
        input_shape = input_shape[:-1] + (input_shape[-1] + hidden_size,)
        output_shape, params = Dense(4 * hidden_size, W_init, b_init)[0](
            rng, input_shape
        )
        # output_shape = input_shape[:-1] + (hidden_size,)
        return output_shape, params

    def apply(params, inputs, prev_state=initial_state()):
        W, b = params
        xh = jnp.concatenate([inputs, prev_state.h], axis=-1)
        gated = jnp.matmul(xh, W) + b
        i, f, o, g = jnp.split(gated, indices_or_sections=4, axis=-1)
        c = sigmoid(f) * prev_state.c + sigmoid(i) * jnp.tanh(g)
        h = sigmoid(o) * jnp.tanh(c)
        return h, LSTMState(h, c)

    return (init, apply)


@module
def LSTM(
    n_layers,
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    Wh_initial_state=zeros,
    bh_initial_state=zeros,
):
    cell = LSTMCell(hidden_size, W_init, b_init, Wh_initial_state, bh_initial_state)
    return Rnn(cell, n_layers)
