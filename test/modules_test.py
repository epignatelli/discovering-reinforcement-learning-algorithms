import jax
import jax.numpy as jnp
from lpg.modules import LSTM, LSTMCell, LSTMState, Rnn


def test_lstm_cell():
    rng = jax.random.PRNGKey(0)
    input_shape = (8,)
    lstm = LSTMCell(16)
    x = jax.random.normal(rng, input_shape)
    out_shape, params = lstm.init(rng, input_shape)
    state = lstm.initial_state(input_shape)
    print(x.shape, state.h.shape, state.c.shape)
    outputs, state = lstm.apply(params, x, state)
    print(outputs.shape, state.h.shape, state.c.shape)


# test_lstm_cell()


def test_lstm():
    rng = jax.random.PRNGKey(0)
    input_shape = (8,)
    lstm = LSTM(5, 16)
    x = jax.random.normal(rng, input_shape)
    out_shape, params = lstm.init(rng, input_shape)
    outputs = lstm.apply(params, x)


test_lstm()