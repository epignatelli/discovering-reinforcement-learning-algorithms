from typing import NamedTuple

import dm_env
from bsuite.baselines import base
from jax.experimental.stax import (
    Dense,
    FanInConcat,
    Identity,
    serial,
    parallel,
    Relu,
    FanOut,
)

from .modules import DiscardHidden, LSTMCell
from .base import module


class HParams(NamedTuple):
    hidden_size: int = 256


@module
def Lpg(hparams):
    phi = serial(Dense(16), Dense(1))
    return serial(
        # FanOut(6),
        parallel(Identity, Identity, Identity, Identity, phi, phi),
        FanInConcat(),
        LSTMCell(hparams.hidden_size)[0:2],
        DiscardHidden(),
        Relu,
        FanOut(2),
        parallel(phi, phi),
    )


class A2C(base.Agent):
    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        return super().select_action(timestep)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        return super().update(timestep, action, new_timestep)
