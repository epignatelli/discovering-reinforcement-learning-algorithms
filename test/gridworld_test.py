from lpg.environments.gridworld import TabularGridworld, RandomGridworld, GridMaps


def test_tabular_gridworld():
    env = TabularGridworld(GridMaps.DENSE, False)
    env.render()
    print(env.reset())
    print(env.step(1))
    env.render()


def test_random_gridworld():
    env = RandomGridworld(GridMaps.SMALL_SPARSE, False)
    env.render()
    print(env.reset())
    print(env.step(1))
    env.render()
