[![Tests](https://github.com/ovgu-FINken/polygonal_roadmaps/actions/workflows/tests.yaml/badge.svg)](https://github.com/ovgu-FINken/polygonal_roadmaps/actions/workflows/tests.yaml)

# polygonal_roadmaps Python package for creating Polygonal Roadmaps and planning paths in said roadmaps
The content of the package is subdivided into three modules: pathfinding, geometry.

## Install
To use the package run `pip install -r requirements.txt` or install the dependencies via conda.

## Pathfinding
The pathfinding package contains all methods realated to pathfinding within the roadmap graph.

## Geometry
The geometry module contains functionality regarding the construction of the reoadmap graph and the geometric considerations used for planning a path in the roadmap.

## Testing
You can execute tests with pytest or push to github and see the result of the github-action.

## Citations
* Roadmap: *S. Mai, M. Deubel, and S. Mostaghim, “Multi-Objective Roadmap Optimization for Multiagent Navigation,” in 2022 IEEE Congress on Evolutionary Computation (CEC), Jul. 2022, pp. 1–8. doi: 10.1109/CEC55065.2022.9870300*.
* CCR Planner: *S. Mai and S. Mostaghim, “Collective Decision-Making for Conflict Resolution in Multi-Agent Pathfinding,” in Swarm Intelligence, Cham, 2022, pp. 79–90. doi: 10.1007/978-3-031-20176-9_7*.
