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
    jax.jit(lstm.apply)(params, x)
    jax.grad(lambda l: sum(lstm.apply(params, x)[0]))(1.0)


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
    jax.jit(lstm.apply)(params, x)
    jax.grad(lambda l: sum(sum(lstm.apply(params, x)[0])))(1.0)


test_lstm()


def test_gru_cell():
    rng = jax.random.PRNGKey(0)
    input_shape = (8,)
    gru = LSTMCell(16)
    x = jax.random.normal(rng, input_shape)
    out_shape, params = gru.init(rng, input_shape)
    outputs, state = gru.apply(params, x)
    print(outputs.shape, state.h.shape, state.c.shape)
    jax.jit(gru.apply)(params, x)
    jax.grad(lambda l: sum(gru.apply(params, x)[0]))(1.0)


test_gru_cell()


def test_gru():
    rng = jax.random.PRNGKey(0)
    SEQ_LEN = 5
    INPUT_FEATURES = 8
    HIDDEN_SIZE = 16
    input_shape = (SEQ_LEN, INPUT_FEATURES)
    gru = LSTM(HIDDEN_SIZE)
    x = jax.random.normal(rng, input_shape)
    out_shape, params = gru.init(rng, input_shape)
    outputs, hidden_state = gru.apply(params, x)
    print(outputs.shape)
    jax.jit(gru.apply)(params, x)
    jax.grad(lambda l: sum(sum(gru.apply(params, x)[0])))(1.0)


test_gru()