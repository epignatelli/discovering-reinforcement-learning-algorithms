import enum
import random
from abc import abstractmethod
from typing import Any, List, NamedTuple, Tuple

import dm_env
import numpy as onp
from dm_env import specs


Point = Tuple[int, int]


class GridworldObject(NamedTuple):
    n: int
    reward: float
    eps_term: float
    eps_respawn: float
    symbol: chr


class GridworldConfig(NamedTuple):
    art: List[str]
    objects: Tuple[GridworldObject]
    max_steps: int
    discount: float = 0.99


class Actions(enum.IntEnum):
    NONE = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4
    SOUTHWEST = 5
    SOUTHEAST = 6
    NORTHEAST = 7
    NORTHWEST = 8

    def vector(self):
        return (
            (0, 0),
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
            (1, -1),
            (1, 1),
            (-1, 1),
            (-1, -1),
        )[int(self)]


class Gridworld(dm_env.Environment):
    def __init__(
        self,
        game_config: GridworldConfig,
        seed: int,
    ):
        # public:
        self.art = onp.array([list(x) for x in game_config.art])
        self.objects = game_config.objects
        self.max_steps = game_config.max_steps
        self.discount = game_config.discount
        self.shape = (len(game_config.art), len(game_config.art[0]))

        # private:
        self._iteration = 0
        self._object_locations = {}
        onp.random.seed(seed)
        random.seed(seed)

    @abstractmethod
    def observation(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def observation_spec(self) -> specs.Array:
        raise NotImplementedError

    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(9, name="action")

    def reset(self) -> dm_env.TimeStep:
        # spawn agent at random location
        self.spawn("P")
        # spawn objects at random location
        for object in self.objects:
            self.spawn(object.symbol, object.n)
        return dm_env.TimeStep(
            dm_env.StepType.FIRST, None, self.discount, self.observation()
        )

    def step(self, action: int) -> dm_env.TimeStep:
        # update agent
        reward = self.act(action)
        # env is stochastic, let it be ❤
        step_type = self.forward()
        # and return the current observation
        return dm_env.TimeStep(step_type, reward, self.discount, self.observation())

    def random_point(self) -> Point:
        return (
            onp.random.randint(0, self.shape[0]),
            onp.random.randint(0, self.shape[1]),
        )

    def empty_point(self) -> Point:
        location = self.random_point()
        while self.art[location] != " ":
            location = self.random_point()
        return location

    def spawn(self, symbol: chr, n: int = 1) -> None:
        for i in range(n):
            location = self.empty_point()
            self.art[location] = symbol
        return

    def locate(self, symbol) -> List[Point]:
        return onp.where(self.art == symbol)

    def reward(self, position: Point) -> float:
        obj = [x for x in self.objects if x.symbol == self.art[position]]
        return obj[0].reward if len(obj) > 0 else 0.0

    def act(self, action: int) -> float:
        agent = self.locate("P")
        reward = 0.0
        # move
        vector = Actions(action).vector()
        location = (
            max(0, min(agent[0] + vector[0], self.shape[0])),
            max(0, min(agent[1] + vector[1], self.shape[1])),
        )
        # hit a wall
        if self.art[location] == "#":
            location = agent

        # stepped on object
        if self.art[location] in [obj.symbol for obj in self.objects]:
            reward = self.reward(location)

        # update agent position
        self.art[agent] = " "
        self.art[location] = "P"
        return reward

    def forward(self) -> dm_env.StepType:
        for obj in self.objects:
            missing = obj.n - len(self.locate(obj.symbol))
            for _ in range(missing):
                #  termination probability
                if onp.random.random() < obj.eps_term:
                    return dm_env.StepType.LAST
                #  respawning probability
                if onp.random.random() < obj.eps_respawn:
                    self.spawn(obj.symbol)
        return dm_env.StepType.MID

    def render(self, mode="human") -> None:
        print()
        print(self.art)

    def seed(self, s: int):
        onp.random.seed(s)
        random.seed(s)
        return


class TabularGridworld(Gridworld):
    def __init__(self, game_config: GridworldConfig, seed: int = 0):
        super().__init__(game_config, seed=seed)
        self._fix_object_locations()

    def observation(self) -> Any:
        i, j = self.locate("P")
        return i * self.shape[0] + j

    def observation_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(sum(self.shape), name="observation")

    def _fix_object_locations(self) -> None:
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


SMALL = GridworldConfig(
    art=[
        "#########",
        "#       #",
        "#  #    #",
        "#       #",
        "#    #  #",
        "#       #",
        "#########",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (2, 1.0, 0.0, 0.05, "a"),
                (2, -1.0, 0.5, 0.1, "b"),
            ],
        )
    ),
    max_steps=500,
)


SMALL_SPARSE = GridworldConfig(
    art=[
        "#########",
        "#       #",
        "#       #",
        "#       #",
        "#       #",
        "#       #",
        "#########",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (1, 1.0, 1.0, 1.0, "a"),
                (2, -1.0, 1.0, 1.0, "b"),
            ],
        )
    ),
    max_steps=50,
)


VERY_DENSE = GridworldConfig(
    art=[
        "#############"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#############"
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (1, 1.0, 0.0, 1.0, "a"),
            ],
        )
    ),
    max_steps=2000,
)


class GridMaps(NamedTuple):
    DENSE = DENSE
    SPARSE = SPARSE
    LONG_HORIZON = LONG_HORIZON
    LONGER_HORIZON = LONGER_HORIZON
    LONG_DENSE = LONG_DENSE
    SMALL = SMALL
    SMALL_SPARSE = SMALL_SPARSE
    VERY_DENSE = VERY_DENSE
