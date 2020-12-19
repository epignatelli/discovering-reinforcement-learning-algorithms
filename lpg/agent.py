from typing import NamedTuple, Tuple

import dm_env
import jax
import jax.numpy as jnp
from bsuite.baselines import base
from bsuite.baselines.utils.sequence import Trajectory
from jax.experimental.optimizers import Optimizer, OptimizerState, adam
from jax.experimental.stax import Dense, FanOut, Flatten, Identity, parallel, serial

from .base import Module, Params, inject, module
from .modules import LSTMCell, LSTMState, ReplayBuffer


class HParams(NamedTuple):
    n_actions: int = 9
    hidden_size: int = 256
    prediction_size: int = 30
    replay_memory_size = 32
    lr: float = 3e-3
    seed: int = 0


class Lpg(base.Agent):
    def __init__(self, hparams):
        @module
        def network() -> Module:
            return serial(
                Flatten,
                serial(Dense(hparams.hidden_size), Dense(hparams.hidden_size)),
                LSTMCell(hparams.hidden_size),
                parallel(FanOut(2), Identity),
                parallel(parallel(Dense(hparams.n_actions), Dense(1)), Identity),
            )

        @inject(self, static_argnums=(0,))
        def forward(
            model: Module,
            params: Params,
            trajectory: Trajectory,
            prev_state: LSTMState,
        ) -> jnp.ndarray:
            outputs = self.model.apply(trajectory.observations, prev_state)
            return loss, (outputs)

        @inject(self, static_argnums=(0, 1))
        def sgd_step(
            model: Module,
            optimiser: Optimizer,
            iteration: int,
            optimiser_state: OptimizerState,
            trajectory: jnp.ndarray,
            prev_state: LSTMState,
        ) -> Tuple[float, OptimizerState]:
            params = optimiser.params_fn(optimiser_state)
            grads, (loss, outputs) = jax.grad(forward, has_aux=True)(
                model, params, trajectory, prev_state
            )
            optimiser_state = optimiser.update_fn(iteration, grads, optimiser_state)
            return loss, optimiser_state

        # public:
        self.hparams = hparams
        self.model = network()
        self.optimiser = Optimizer(*adam(hparams.lr))
        self.iteration = 0
        self.buffer = ReplayBuffer(hparams.replay_memory_size)

        # private:
        input_shape = (1, 1 + 1 + 1 + 1 + hparams.prediction_size * 2)
        self._rng = jax.random.PRNGKey(hparams.seed)
        self._params = self.model.init(self._rng, input_shape)
        self._optimiser_state = self.optimiser.init_fn(self._params)
        self._prev_state = None

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        (logits, _), self._prev_state = self.forward(
            self.params, timestep.observation, self._prev_state
        )
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

        #  learn only if full
        if self.buffer.full() or new_timestep.last():
            trajectory = self.buffer.sample()
            loss, self._optimiser_state = self.sgd_step(
                self.model,
                self.optimiser,
                self.iteration,
                self._optimiser_state,
                trajectory,
                self._prev_state,
            )

        if new_timestep.last():
            self._prev_state = None

        return
