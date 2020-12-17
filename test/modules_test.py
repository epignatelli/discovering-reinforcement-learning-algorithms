import jax
import jax.numpy as jnp
from lpg.modules import LSTM, LSTMCell, LSTMState, Rnn


def test_lstm_cell():
    rng = jax.random.PRNGKey(0)
    input_shape = (8,)
    lstm = LSTMCell(16)
    x = jax.random.normal(rng, input_shape)
    out_shape, params = lstm.init(rng, input_shape)
    outputs, state = lstm.apply(params, x)
    print(outputs.shape, state.h.shape, state.c.shape)


test_lstm_cell()


def test_lstm():
    rng = jax.random.PRNGKey(0)
    SEQ_LEN = 5
    INPUT_FEATURES = 8
    HIDDEN_SIZE = 16
    input_shape = (SEQ_LEN, INPUT_FEATURES)
    lstm = LSTM(HIDDEN_SIZE)
    x = jax.random.normal(rng, input_shape)
    out_shape, params = lstm.init(rng, input_shape)
    outputs, hidden_state = lstm.apply(params, x)
    print(outputs.shape)


test_lstm()