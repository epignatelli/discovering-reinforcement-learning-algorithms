import jax
import jax.numpy as jnp
from lpg.modules import LSTM, LSTMCell, LSTMState, Rnn


def test_lstm_cell():
    rng = jax.random.PRNGKey(0)
    input_shape = (8,)
    lstm = LSTMCell(16)
    x = jax.random.normal(rng, input_shape)
    out_shape, params = lstm.init(rng, input_shape)
    outputs = lstm.apply(params, x)


def test_lstm():
    rng = jax.random.PRNGKey(0)
    input_shape = (5, 8)
    lstm = LSTM(5, 16)
    x = jax.random.normal(rng, input_shape)
    out_shape, params = lstm.init(rng, input_shape)
    outputs = lstm.apply(params, x)


test_lstm_cell()