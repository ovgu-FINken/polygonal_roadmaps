from polygonal_roadmaps import pathfinding
from polygonal_roadmaps import geometry
import networkx as nx

import numpy as np
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
    def __init__(self, map_file, scenario_file, scenario_index=0, n_agents=None) -> None:
        graph, start, goal = None, None, None
        graph = pathfinding.read_movingai_map(map_file)
        df = pd.read_csv(scenario_file, sep="\t", names=["id", "map", "w", "h", "x0", "y0", "x1", "y1", "cost"], skiprows=1)
        sg = df.loc[df.id.eq(scenario_index), "x0":"y1"].to_records(index=False)
        if n_agents is None:
            n_agents = len(sg)
        start = ((x, y) for x, y, *_ in sg[:n_agents])
        goal = ((x, y) for *_, x, y, in sg[:n_agents])
        super().__init__(graph, start, goal)


class RoadmapEnvironment(Environment):
    def __init__(self, map, start_positions, goal_positions, grid):
        pass


class Planner():
    def __init__(self, environment) -> None:
        self.env = environment


class FixedPlanner(Planner):
    def __init__(self, environment, plan) -> None:
        super().__init__(environment)
        self._plan = plan

    def get_plan(self, *_):
        return self._plan


class Executor():
    def __init__(self, environment: Environment, planner: Planner, time_frame: int = 100) -> None:
        self.env = environment
        self.planner = planner
        self.history = [self.env.state]
        self.time_frame = time_frame

    def run(self, update=False):
        if len(self.history) >= self.time_frame:
            return
        plan = self.planner.get_plan(self.env)
        for _ in range(self.time_frame):
            self.step(plan)
            if all(s is None for s in self.env.state):
                return self.history
            if update:
                plan = self.planner.get_plan(self.env)
            else:
                plan = plan[1:]

    def plan(self, update=False):
        plan = self.planner.get_plan(self.env, self.env.state)
        self.step(plan)
        self.run(update=True)

    def step(self, plan):
        # advance agents
        state = list(plan[1])
        for i, s in enumerate(self.env.goal):
            if state[i] == s:
                state[i] = None
        self.env.state = tuple(state)
        self.history.append(self.env.state)
        return self.env.state


def main():
    environment = Environment()
    planner = Planner(environment)
    executor = Executor(environment, planner)
    executor.run()
    print(executor.history[-1])


if __name__ == "__main__":
    main()
