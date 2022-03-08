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


class Executor():
    def __init__(self, environment: Environment, planner: Planner) -> None:
        self.env = environment
        self.planner = planner
        
    def run(self, update=False):
        plan = self.planner.plan(self.env)
        if not update:
            return plan
        elif self.env.state == self.env.goal:
            return
        else:
            self.step(plan)
            self.run(update=True)

    def step(self):
        # advance agents
        pass


def main():
    environment = Environment()
    planner = Planner(environment)
    executor = Executor(environment, planner)
    print(executor)


if __name__ == "__main__":
    main()
