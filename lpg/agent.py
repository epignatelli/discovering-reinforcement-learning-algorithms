from typing import NamedTuple

import dm_env
from bsuite.baselines import base
import jax
from jax.experimental.stax import (
    Dense,
    FanInConcat,
    Identity,
    serial,
    parallel,
    Relu,
    FanOut,
    Flatten,
)
from jax.experimental.optimizers import adam

from .modules import DiscardHidden, LSTMCell
from .base import module


class HParams(NamedTuple):
    hidden_size: int = 256
    lr: float = 3e-3
    seed: int = 0


@module
def Phi():
    return serial(Dense(16), Dense(1))


@module
def Lpg(hparams):
    phi = serial(Dense(16), Dense(1))
    return serial(
        parallel(Identity, Identity, Identity, Identity, phi, phi),
        FanInConcat(),
        LSTMCell(hparams.hidden_size)[0:2],
        DiscardHidden(),
        Relu,
        FanOut(2),
        parallel(phi, phi),
    )


class A2C(base.Agent):
    def __init__(self, hparams):
        @module
        def network():
            h = hparams.hidden_size
            return serial(
                Flatten(),
                serial(Dense(h), Dense(h)),
                LSTMCell(hparams.hidden_size),
                parallel(Dense(hparams.n_actions), Identity),
                parallel(Dense(1), Identity),
            )

        def loss():
            pass

        def sgd_step(
            model,
            optimiser,
            iteration,
            optimiser_state,
        ):
            pass

        # public:
        self.hparams = hparams
        self.model = network()
        self.optimiser = adam(hparams.lr)
        self.iteration = 0
        self.params = self.model.init()
        # private:
        self._rng = jax.random.PRNGKey(hparams.seed)
        self.configure_optimiser()

    def configure_optimiser(self):
        self.optimiser = adam(self.hparams.lr)
        self._optimiser_state = self.optimiser.init_fn(self.params)
        return

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        return super().select_action(timestep)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        return super().update(timestep, action, new_timestep)
