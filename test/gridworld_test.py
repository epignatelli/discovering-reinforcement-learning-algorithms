from lpg import base
from lpg.environments import TabularGridworld, RandomGridworld, DENSE


def test_tabular_gridworld():
    env = TabularGridworld(DENSE, False)
    print(env.reset())
    print(env.step(1))


def test_random_gridworld():
    env = RandomGridworld(DENSE, False)
    print(env.reset())
    print(env.step(1))


# test_tabular_gridworld()
# test_random_gridworld