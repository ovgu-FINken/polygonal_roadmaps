from polygonal_roadmaps import pathfinding
from polygonal_roadmaps import geometry

import numpy as np


class PolygonalRoadmap:
    def __init__(self, generators: np.array, map=None, x_min=0, x_max=1, y_min=0, y_max=1):
        self.free, self.occupied = geometry.read_obstacles(map)
        self.graph = geometry.create_graph(
            generators,
            working_area_x=(x_min, x_max),
            working_area_y=(y_min, y_max),
            occupied_space=self.occupied)


def main():
    roadmap = PolygonalRoadmap(
        geometry.square_tiling(0.5, working_area_x=(-1, 3), working_area_y=(-1, 3)),
        map='test/resources/icra_2021_map.yaml',
        x_min=-1,
        x_max=3,
        y_min=-1,
        y_max=3)


if __name__ == "__main__":
    main()
