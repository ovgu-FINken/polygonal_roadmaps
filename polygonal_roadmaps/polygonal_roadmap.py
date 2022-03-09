from polygonal_roadmaps import pathfinding
from polygonal_roadmaps import geometry
from itertools import zip_longest
import networkx as nx
from pathlib import Path
from cProfile import Profile

import pandas as pd


class Environment():
    def __init__(self, graph: nx.Graph, start: tuple, goal: tuple) -> None:
        self.g = graph
        self.state = start
        self.start = start
        self.goal = goal

    def get_graph(self) -> nx.Graph:
        return self.g


class GraphEnvironment(Environment):
    def __init__(self, graph: nx.Graph, start: tuple, goal: tuple) -> None:
        super().__init__(graph, start, goal)


class MapfInfoEnvironment(Environment):
    def __init__(self, map_file, scenario_file, n_agents=None) -> None:
        graph, start, goal = None, None, None
        graph = pathfinding.read_movingai_map(map_file)
        df = pd.read_csv(scenario_file, sep="\t", names=["id", "map", "w", "h", "x0", "y0", "x1", "y1", "cost"], skiprows=1)
        sg = df.loc[:, "x0":"y1"].to_records(index=False)
        if n_agents is None:
            n_agents = len(sg)
        start = [(x, y) for y, x, *_ in sg[:n_agents]]
        goal = [(x, y) for *_, y, x, in sg[:n_agents]]
        super().__init__(graph, start, goal)


class RoadmapEnvironment(Environment):
    def __init__(self, map, start_positions, goal_positions, grid):
        self.g = geometry.create_graph(None)


class Planner():
    def __init__(self, environment) -> None:
        self.env = environment


class FixedPlanner(Planner):
    def __init__(self, environment, plan) -> None:
        super().__init__(environment)
        self._plan = plan

    def get_plan(self, *_):
        return self._plan


class CBSPlanner(Planner):
    def __init__(self, environment, **kwargs) -> None:
        super().__init__(environment)
        sg = list(zip(self.env.state, self.env.goal))
        self.cbs = pathfinding.CBS(self.env.g, sg, **kwargs)

    def get_plan(self, *_):
        self.cbs.run()
        plans = list(self.cbs.best.solution)
        return list(zip_longest(*plans)) + [(None for _ in plans)]


class Executor():
    def __init__(self, environment: Environment, planner: Planner, time_frame: int = 100) -> None:
        self.env = environment
        self.planner = planner
        self.history = [self.env.state]
        self.time_frame = time_frame
        self.profile = Profile()

    def run(self, update=False):
        if len(self.history) >= self.time_frame:
            return
        self.profile.enable()
        plan = self.planner.get_plan(self.env)
        for _ in range(self.time_frame):
            self.step(plan)
            if all(s is None for s in self.env.state):
                plan = plan[1:]
                self.profile.disable()
                return self.history
            if update:
                plan = self.planner.get_plan(self.env)
            else:
                plan = plan[1:]
        self.profile.disable()

    def step(self, plan):
        # advance agents
        state = list(plan[1])
        for i, s in enumerate(self.env.goal):
            if state[i] == s:
                state[i] = None
        self.env.state = tuple(state)
        self.history.append(self.env.state)
        return self.env.state


def make_run(map_path=None, scen_path=None, n_agents=2):
    if map_path is None:
        map_path = Path() / "test" / "resources" / "random-32-32-10.map"
    if scen_path is None:
        scen_path = Path() / "test" / "resources" / "random-32-32-10-even-1.scen"
    env = MapfInfoEnvironment(map_path, scen_path, n_agents=n_agents)
    planner = CBSPlanner(env, limit=100)
    executor = Executor(env, planner)
    executor.run(update=False)
    executor.profile.print_stats(sort=2)
    print(f"steps in history: {len(executor.history)}")
    return executor


if __name__ == "__main__":
    make_run()
