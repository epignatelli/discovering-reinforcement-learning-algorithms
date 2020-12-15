from typing import Any, NamedTuple, List, Tuple
import enum

import numpy as onp
from dm_env import specs

from ..base import Gridworld, GridworldConfig, GridworldObject


DENSE = GridworldConfig(
    art=[
        "#############",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#############",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (2, 1.0, 0.0, 0.05, "a"),
                (1, -1.0, 0.5, 0.1, "b"),
                (1, -1.0, 0.0, 0.5, "c"),
            ],
        )
    ),
    max_steps=500,
)

SPARSE = GridworldConfig(
    art=[
        "###############",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "###############",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [(1, 1.0, 1.0, 0.0, "a"), (1, -1.0, 1.0, 0.0, "b")],
        )
    ),
    max_steps=50,
)


LONG_HORIZON = GridworldConfig(
    art=[
        "#############",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#############",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [(2, 1.0, 0.0, 0.01, "a"), (2, -1.0, 0.5, 1.0, "b")],
        )
    ),
    max_steps=1000,
)

LONGER_HORIZON = GridworldConfig(
    art=[
        "###########",
        "#    #    #",
        "#         #",
        "#    #    #",
        "#   ###   #",
        "#    #    #",
        "#         #",
        "#    #    #",
        "###########",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (2, 1.0, 0.1, 0.01, "a"),
                (5, -1.0, 0.8, 1.0, "b"),
            ],
        )
    ),
    max_steps=2000,
)

LONG_DENSE = GridworldConfig(
    art=[
        "#############",
        "#           #",
        "#     #     #",
        "#     #     #",
        "#     #     #",
        "### ##### ###",
        "#     #     #",
        "#     #     #",
        "#           #",
        "#     #     #",
        "#     #     #",
        "#     #     #",
        "### ##### ###",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (4, 1.0, 0.0, 0.005, "a"),
            ],
        )
    ),
    max_steps=2000,
)


class RandomGridworld(Gridworld):
    def __init__(self, game_config: GridworldConfig, seed: int = 0):
        super().__init__(game_config, seed=seed)
        self._set_object_locations()

    def observation(self) -> Any:
        return onp.stack([onp.where(self.art == x.symbol, 1, 0) for x in self.objects])

    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray((len(self.objects), *self.shape), name="observation")

    def _set_object_locations(self) -> None:
        self._object_locations = {obj.symbol: [] for obj in self.objects}
        for obj in self.objects:
            for _ in range(obj.n):
                self._object_locations[obj.symbol].append(self.empty_point())
        return

    def spawn(self, symbol, n=1) -> None:
        if symbol in self._object_locations:
            for location in self._object_locations[symbol]:
                self.art[location] = symbol
        else:
            self.art[self.empty_point()] = symbol
        return
