from attr import attributes
from polygonal_roadmaps import pathfinding
from polygonal_roadmaps import geometry
import networkx as nx

import numpy as np


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
