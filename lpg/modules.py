from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal, zeros

from .base import Module, module


@module
def Rnn(cell):
    """Layer construction function for an RNN unroll.
    Implements truncated backpropagation through time for the backward pass
    """

    def init(rng, input_shape):
        return cell.init(rng, input_shape)

    def apply(params, inputs, **kwargs):
        prev_state = kwargs.pop("prev_state", cell.initial_state())
        prev_state, outputs = jax.lax.scan(
            lambda prev_state, inputs: cell.apply(
                params, inputs, prev_state=prev_state
            )[::-1],
            prev_state,
            inputs,
        )
        return outputs, prev_state

    return (init, apply)


@module
def DiscardHidden():
    def init(rng, input_shape):
        return input_shape, ()

    def apply(params, inputs, **kwargs):
        return inputs[0]

    return init, apply


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
    Formulation: Zaremba, W., 2015, https://arxiv.org/pdf/1409.2329.pdf"""

    def initial_state():
        shape = (hidden_size,)
        k1, k2 = jax.random.split(jax.random.PRNGKey(initial_state_seed))
        return LSTMState(h_initial_state_fn(k1, shape), c_initial_state_fn(k2, shape))

    def init(rng, input_shape):
        in_dim, out_dim = input_shape[-1] + hidden_size, 4 * hidden_size
        output_shape = input_shape[:-1] + (hidden_size,)
        k1, k2 = jax.random.split(rng)
        W, b = W_init(k1, (in_dim, out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply(params, inputs, **kwargs):
        prev_state = kwargs.pop("prev_state", initial_state())
        W, b = params
        xh = jnp.concatenate([inputs, prev_state.h], axis=-1)
        gated = jnp.matmul(xh, W) + b
        i, f, o, g = jnp.split(gated, indices_or_sections=4, axis=-1)
        c = sigmoid(f) * prev_state.c + sigmoid(i) * jnp.tanh(g)
        h = sigmoid(o) * jnp.tanh(c)
        return h, LSTMState(h, c)

    return (init, apply, initial_state)


@module
def LSTM(
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    h_initial_state=zeros,
    c_initial_state=zeros,
    initial_state_seed=0,
):
    return Rnn(
        LSTMCell(
            hidden_size,
            W_init,
            b_init,
            h_initial_state,
            c_initial_state,
            initial_state_seed,
        )
    )


@module
def GRUCell(
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    initial_state_fn=zeros,
    initial_state_seed=0,
):
    """Layer construction function for an GRU cell.
    Formulation: Chun, J., 2014, https://arxiv.org/pdf/1412.3555v1.pdf"""

    def initial_state():
        rng = jax.random.PRNGKey(initial_state_seed)
        return initial_state_fn(rng, (hidden_size,))

    def init(rng, input_shape):
        in_dim, out_dim = input_shape[-1] + hidden_size, 3 * hidden_size
        output_shape = input_shape[:-1] + (hidden_size,)
        k1, k2 = jax.random.split(rng)
        W_i = W_init(k1, (in_dim, out_dim))
        W_h = W_init(k1, (in_dim, out_dim))
        b = b_init(k1, (out_dim,))
        return output_shape, (W_i, W_h, b)

    def apply(params, inputs, **kwargs):
        prev_state = kwargs.pop("prev_state", initial_state())
        W_i, W_h, b = params
        W_hz, W_ha = jnp.split(W_h, indices_or_sections=(2 * hidden_size,), axis=-1)
        b_z, b_a = jnp.split(b, indices_or_sections=(2 * hidden_size,), axis=-1)

        gated = jnp.matmul(inputs, W_i)
        zr_x, a_x = jnp.split(gated, indices_or_sections=[2 * hidden_size], axis=-1)
        zr_h = jnp.matmul(prev_state, W_hz)
        z, r = jnp.split(
            jax.nn.sigmoid(zr_x + zr_h + b_z), indices_or_sections=2, axis=-1
        )
        a_h = jnp.matmul(r * prev_state, W_ha)
        a = jnp.tanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))
        h = (1 - z) * prev_state + z * a
        return h, h

    return (init, apply, initial_state)


@module
def GRU(
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    initial_state_fn=zeros,
    initial_state_seed=0,
):
    return Rnn(
        GRUCell(hidden_size, W_init, b_init, initial_state_fn, initial_state_seed)
    )
