import dm_env
from bsuite.baselines import base

from .modules import LSTM


def HParams(NamedTuple):
    n_layers: int
    hidden_size: int


def Lpg(hparams):
    return LSTM(hparams.n_layers, hparams.hidden_size)


class ActorCritic(base.Agent):
    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        return super().select_action(timestep)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        return super().update(timestep, action, new_timestep)
