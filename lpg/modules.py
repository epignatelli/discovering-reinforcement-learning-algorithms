from typing import NamedTuple

import jax
from jax._src.numpy.lax_numpy import __array_module__
import jax.numpy as jnp
from jax.experimental.stax import Dense
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal, zeros

from .base import module, RnnCell, rnn_cell


@module
def Rnn(cell: RnnCell, n_layers: int):
    def init(rng, input_shape):
        return cell.init(rng, input_shape)

    def apply(params, inputs, prev_state=None):
        if prev_state is None:
            prev_state = cell.initial_state(inputs.shape)
        outputs, prev_state = cell.apply(params, inputs, prev_state)
        outputs, prev_state = jax.lax.fori_loop(
            0,
            n_layers,
            lambda i, val: cell.apply(params, val[0], val[1]),
            (outputs, prev_state),
        )
        return outputs, prev_state

    return (init, apply)


class LSTMState(NamedTuple):
    h: jnp.ndarray
    c: jnp.ndarray


@rnn_cell
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

    def initial_state(input_shape):
        shape = input_shape[:-1] + (hidden_size,)
        k1, k2 = jax.random.split(jax.random.PRNGKey(initial_state_seed))
        return LSTMState(h_initial_state_fn(k1, shape), c_initial_state_fn(k2, shape))

    def init(rng, input_shape):
        input_shape = input_shape[:-1] + (input_shape[-1] + hidden_size,)
        output_shape, params = Dense(4 * hidden_size, W_init, b_init)[0](
            rng, input_shape
        )
        return output_shape, params

    def apply(params, inputs, prev_state):
        W, b = params
        xh = jnp.concatenate([inputs, prev_state.h], axis=-1)
        gated = jnp.matmul(xh, W) + b
        i, f, o, g = jnp.split(gated, indices_or_sections=4, axis=-1)
        c = sigmoid(f) * prev_state.c + sigmoid(i) * jnp.tanh(g)
        h = sigmoid(o) * jnp.tanh(c)
        return h, LSTMState(h, c)

    return (initial_state, init, apply)


@module
def LSTM(
    n_layers,
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    h_initial_state=zeros,
    c_initial_state=zeros,
    initial_state_seed=0,
):
    cell = LSTMCell(
        hidden_size,
        W_init,
        b_init,
        h_initial_state,
        c_initial_state,
        initial_state_seed,
    )
    return Rnn(cell, n_layers)
