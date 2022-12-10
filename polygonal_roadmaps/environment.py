import networkx as nx
import logging
import pandas as pd
import numpy as np
from polygonal_roadmaps import geometry, planner

from pathlib import Path


def gen_example_graph(a, b):
    if b > a:
        a, b = b, a
    example = nx.Graph()
    for i in range(a):
        example.add_node(chr(i + ord('a')), pos=(i, 0))
    for i in range(a - 1):
        example.add_edge(chr(i + ord('a')), chr(i + ord('b')))
    for i in range(a, a + b):
        example.add_node(chr(i + ord('a')), pos=(i - a + (a - b) / 2, 1))
    for i in range(a, a + b - 1):
        example.add_edge(chr(i + ord('a')), chr(i + ord('b')))
    example.add_edge(chr(ord('a') + int((a - b) / 2)), chr(ord('a') + a))
    example.add_edge(chr(ord('a') - int((a - b) / 2) - 1 + a), chr(ord('a') + a + b - 1))

    for source, sink in example.edges():
        p1 = np.array(example.nodes()[source]['pos'])
        p2 = np.array(example.nodes()[sink]['pos'])
        example.edges()[source, sink]['dist'] = np.linalg.norm(p1 - p2)

    return example


class Environment():
    def __init__(self, graph: nx.Graph, start: tuple, goal: tuple) -> None:
        self.g = graph
        self.state = start
        self.start = start
        self.goal = goal
        self.perturbate_weights()

    def perturbate_weights(self):
        for u, v in self.g.edges():
            if 'dist' not in self.g.edges[u, v]:
                self.g.edges[u, v]['dist'] = 1.0
            self.g.edges[u, v]['dist'] *= 1 + 0.00001 * np.random.randn()

    def get_graph(self) -> nx.Graph:
        return self.g


class GraphEnvironment(Environment):
    def __init__(self, graph: nx.Graph, start: tuple, goal: tuple) -> None:
        super().__init__(graph, start, goal)


class MapfInfoEnvironment(Environment):
    def __init__(self, scenario_file, n_agents=None) -> None:
        graph, start, goal = None, None, None
        df = pd.read_csv(Path("benchmark") / "scen" / scenario_file,
                         sep="\t",
                         names=["id", "map_name", "w", "h", "x0", "y0", "x1", "y1", "cost"],
                         skiprows=1)
        self.width = df.w[0]
        self.height = df.h[0]
        self.map_file = Path() / "benchmark" / "maps" / df.map_name[0]
        self.scenario_file = scenario_file
        graph, w, h, data = planner.read_movingai_map(self.map_file)

        sg = df.loc[:, "x0":"y1"].to_records(index=False)
        if n_agents is None:
            n_agents = len(sg)
        start = [(x, y) for y, x, *_ in sg[:n_agents]]
        goal = [(x, y) for *_, y, x, in sg[:n_agents]]
        for s in start:
            if s not in graph:
                logging.error(f"start {s} not in map {self.map_file}, value is {data[s[0]][s[1]]}")

        for g in goal:
            if g not in graph:
                logging.error(f"goal {g} not in map {self.map_file}, value is {data[s[0]][s[1]]}")
        super().__init__(graph, start, goal)

    def get_background_matrix(self):
        positions = nx.get_node_attributes(self.g, 'pos').values()
        positions = np.array(list(positions))
        image = np.zeros((self.width, self.height))
        for x, y in nx.get_node_attributes(self.g, 'pos').values():
            image[x, y] = 1
        return image

    def get_obstacle_df(self):
        m = self.get_background_matrix()
        data = []
        for x in range(m.shape[0]):
            for y in range(m.shape[1]):
                if m[x, y] == 0:
                    data.append({'x': x, 'y': y, 'status': 'free'})
                else:
                    data.append({'x': x, 'y': y, 'status': 'occupied'})
        df = pd.DataFrame(data)
        df['status'] = df['status'].astype('category')
        return df


class RoadmapEnvironment(Environment):
    def __init__(self, map_path, start_positions, goal_positions, generator_points=None, wx=None, wy=None, offset=0.15):
        _, obstacles = geometry.read_obstacles(map_path)
        if wx is None:
            wx = (-1, 3.5)
        if wy is None:
            wy = (-3, 1.5)
        if generator_points is None:
            generator_points = geometry.square_tiling(1.0, working_area_x=wx, working_area_y=wy)
        graph = geometry.create_graph(generator_points,
                                      working_area_x=wx,
                                      working_area_y=wy,
                                      offset=offset,
                                      occupied_space=obstacles)
        super().__init__(graph, start_positions, goal_positions)