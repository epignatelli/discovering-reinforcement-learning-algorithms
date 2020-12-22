from functools import partial
from typing import NamedTuple, Tuple

import dm_env
import jax
from bsuite.baselines import base
from bsuite.baselines.utils.sequence import Trajectory
from helx.methods import purify, module
from helx.types import Module, Params
from jax import numpy as jnp
from jax.experimental.optimizers import Optimizer, OptimizerState, adam
from jax.experimental.stax import Dense, FanOut, Flatten, Identity, parallel, serial

from .modules import LSTMCell, LSTMState, ReplayBuffer


class MetaHParams(NamedTuple):
    oprimiser: Optimizer = adam
    learning_rate: float = jax.random.uniform([5e-4, 1e-4, 3e-5])
    discount_factor: float = jax.random.uniform([0.995, 0.99])
    policy_entropy_cost: float = jax.random.uniform([0.01, 0.01])
    prediction_entropy_cost: float = 0.001
    policy_regularisation_weight: float = 0.001
    policy_regularisation_weight: float = 0.001
    bandit_temperature: float = 0.1
    bandit_exploration_bonus: float = 0.2
    trajectory_steps: int = 20
    parameters_update: int = 5
    parallel_lifetimes: int = 960
    parallel_environments: int = 64


class AgentHParams(NamedTuple):
    n_actions: int = 9
    hidden_size: int = 256
    prediction_size: int = 30
    replay_memory_size = 32
    optimiser = adam
    lr: float = 3e-3
    kl_cost: float = 0.5
    seed: int = 0


class Lpg:
    def __init__(self, hparams, backbone, input_shape, seed=0):
        self.hparams = hparams
        self.rng = jax.random.PRNGKey(seed)
        self.optimiser = adam(hparams.lr)
        self.model = serial(
            Flatten,
            backbone,
            LSTMCell(hparams.hidden_size),
            parallel(FanOut(2), Identity),
            parallel(parallel(Dense(hparams.n_actions), Dense(1)), Identity),
        )
        _, self.params = self.model.init(self.rng, input_shape)

    @partial(purify, static_argnums=0)
    def forward(model, optimiser):
        return loss, y_hat

    @partial(purify, static_argnums=0)
    def sgd_step(model, optimiser, iteration, optimiser_state, x):
        return loss, optimiser_state


class A2C(base.Agent):
    def __init__(self, network, optimiser, hparams):
        # public:
        self.hparams = hparams
        self.model = network()
        self.optimiser = optimiser
        self.iteration = 0
        self.buffer = ReplayBuffer(hparams.replay_memory_size)

        # private:
        self._rng = jax.random.PRNGKey(hparams.seed)
        input_shape = (1, 1 + 1 + 1 + 1 + hparams.prediction_size * 2)
        params = self.model.init(self._rng, input_shape)
        self._optimiser_state = self.optimiser.init_fn(params)
        self._prev_state = None

    @partial(purify, static_argnums=(0,))
    def forward(
        model: Module,
        params: Params,
        trajectory: Trajectory,
        prev_state: LSTMState,
    ) -> jnp.ndarray:
        outputs, prev_state = model.apply(trajectory.observations, prev_state)
        return loss, (outputs)

    @partial(purify, static_argnums=(0, 1))
    def sgd_step(
        model: Module,
        optimiser: Optimizer,
        iteration: int,
        optimiser_state: OptimizerState,
        trajectory: jnp.ndarray,
        prev_state: LSTMState,
    ) -> Tuple[float, OptimizerState]:
        params = optimiser.params_fn(optimiser_state)
        grads, (loss, outputs) = jax.grad(A2C.forward, has_aux=True)(
            model, params, trajectory, prev_state
        )
        optimiser_state = optimiser.update_fn(iteration, grads, optimiser_state)
        return loss, optimiser_state

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        (logits, _), prev_state = self.model.apply(
            self.params, timestep.observation, self._prev_state
        )
        self._prev_state = prev_state
        action = jax.random.categorical(self._rng, logits).squeeze()
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        self.iteration += 1

        #  add transtition to memory
        self.buffer.add(timestep, action, new_timestep)

        #  learn only if buffer full or if we need to reset
        if self.buffer.full() or new_timestep.last():
            trajectory = self.buffer.sample()
            loss, optimiser_state = self.sgd_step(
                self.model,
                self.optimiser,
                self.iteration,
                self._optimiser_state,
                trajectory,
                self._prev_state,
            )
            self._optimiser_state = optimiser_state

        # reset hidden state to avoid information flowing across episodes
        if new_timestep.last():
            self._prev_state = None

        return
