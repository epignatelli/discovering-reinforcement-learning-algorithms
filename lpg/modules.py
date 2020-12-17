from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.stax import Dense
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal, zeros

from .base import RnnCell, module, rnn_cell


@module
def Rnn(cell: RnnCell):
    def init(rng, input_shape):
        return cell.init(rng, input_shape)

    def apply(params, inputs, prev_state=cell.initial_state()):
        prev_state, outputs = jax.lax.scan(
            lambda prev_state, inputs: cell.apply(params, inputs, prev_state)[::-1],
            prev_state,
            inputs,
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

    def initial_state():
        shape = (hidden_size,)
        k1, k2 = jax.random.split(jax.random.PRNGKey(initial_state_seed))
        return LSTMState(h_initial_state_fn(k1, shape), c_initial_state_fn(k2, shape))

    def init(rng, input_shape):
        in_dim, out_dim = input_shape[-1] + hidden_size, 4 * hidden_size
        k1, k2 = jax.random.split(rng)
        W, b = W_init(k1, (in_dim, out_dim)), b_init(k2, (out_dim,))
        output_shape = input_shape[:-1] + (hidden_size,)
        return output_shape, (W, b)

    def apply(params, inputs, prev_state=initial_state()):
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
    return Rnn(cell)
