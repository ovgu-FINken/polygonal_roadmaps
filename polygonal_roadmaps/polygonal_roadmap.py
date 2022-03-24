from dataclasses import dataclass
import pstats
import io
from polygonal_roadmaps import pathfinding
from polygonal_roadmaps import geometry
from itertools import zip_longest
import networkx as nx
import numpy as np
from pathlib import Path
from cProfile import Profile

import logging

import pandas as pd


def create_df_from_profile(profile):
    # see https://qxf2.com/blog/saving-cprofile-stats-to-a-csv-file/
    strio = io.StringIO()
    ps = pstats.Stats(profile, stream=strio)
    ps.print_stats(1.0)

    result = strio.getvalue()
    result = 'ncalls' + result.split('ncalls')[-1]
    result = '\n'.join([','.join(line.rstrip().split(None, 5)) for line in result.split('\n')])

    csv = io.StringIO(result)
    df = pd.read_csv(csv)

    def flf(s):
        if s[0] == '{':
            return ('{buildins}', pd.NA, s)
        else:
            parts = s.split(':')
            return (':'.join(parts[:-1]), parts[-1].split('(')[0], '(' + '('.join(parts[-1].split('(')[1:]))

    def extract_file(s):
        return flf(s)[0]

    def extract_line(s):
        return flf(s)[1]

    def extract_function(s):
        return flf(s)[2]

    df['file'] = df["filename:lineno(function)"].apply(extract_file)
    df['line'] = df["filename:lineno(function)"].apply(extract_line)
    df['function'] = df["filename:lineno(function)"].apply(extract_function)
    return df


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
    def __init__(self, scenario_file, n_agents=None) -> None:
        graph, start, goal = None, None, None
        df = pd.read_csv(Path("benchmark") / "scen" / scenario_file, sep="\t", names=["id", "map_name", "w", "h", "x0", "y0", "x1", "y1", "cost"], skiprows=1)
        self.width = df.w[0]
        self.height = df.h[0]
        self.map_file = Path() / "benchmark" / "maps" / df.map_name[0]
        self.scenario_file = scenario_file
        graph = pathfinding.read_movingai_map(self.map_file)

        sg = df.loc[:, "x0":"y1"].to_records(index=False)
        if n_agents is None:
            n_agents = len(sg)
        start = [(x, y) for y, x, *_ in sg[:n_agents]]
        goal = [(x, y) for *_, y, x, in sg[:n_agents]]
        super().__init__(graph, start, goal)

    def get_background_matrix(self):
        positions = nx.get_node_attributes(self.g, 'pos').values()
        positions = np.array(list(positions))
        image = np.zeros((self.width, self.height))
        for x, y in nx.get_node_attributes(self.g, 'pos').values():
            image[x, y] = 1
        return image.transpose()


class RoadmapEnvironment(Environment):
    def __init__(self, map, start_positions, goal_positions, grid):
        self.g = geometry.create_graph(None)


class Planner():
    def __init__(self, environment, replan_required=False) -> None:
        self.env = environment
        self.replan_required = replan_required


class FixedPlanner(Planner):
    def __init__(self, environment, plan) -> None:
        super().__init__(environment)
        self._plan = plan

    def get_plan(self, *_):
        return self._plan


class CBSPlanner(Planner):
    def __init__(self, environment, horizon: int = None, **kwargs) -> None:
        # initialize the planner.
        # if the horizon is not None, we want to replan after execution of one step
        super().__init__(environment, replan_required=(horizon is not None))
        self.kwargs = kwargs
        sg = [(s, g) for s, g in zip(self.env.state, self.env.goal) if s is not None]
        self.cbs = pathfinding.CBS(self.env.g, sg, **self.kwargs)

    def get_plan(self, *_):
        if self.replan_required:
            self.cbs.update_state(self.env.state)
        self.cbs.run()
        plans = list(self.cbs.best.solution)
        # reintroduce plan for those states that have already finished -> i.e., where state is None
        j = 0
        ret = []
        logging.info(f"state: {self.env.state}")
        for i, s in enumerate(self.env.state):
            if s is not None:
                ret.append(plans[j] + [None])
                j += 1
            else:
                ret.append([None])
        ret = zip_longest(*ret, fillvalue=None)
        return list(ret)


def convert_history_to_df(history):
    records = []
    for t, state in enumerate(history):
        for agent, pos in enumerate(state):
            if pos is None:
                continue
            records.append({'agent': agent, 'x': pos[0], 'y': pos[1], 't': t})
    return pd.DataFrame(records)


class Executor():
    def __init__(self, environment: Environment, planner: Planner, time_frame: int = 100) -> None:
        self.env = environment
        self.planner = planner
        self.history = [self.env.state]
        self.time_frame = time_frame
        self.profile = Profile()

    def run(self, profiling=True):
        if len(self.history) >= self.time_frame:
            return
        if profiling:
            self.profile.enable()
        plan = self.planner.get_plan(self.env)
        for i in range(self.time_frame):
            logging.info(f"At iteration {i} / {self.time_frame}")
            self.step(plan)
            # (goal is reached)
            if all(s is None for s in self.env.state):
                # plan = plan[1:]
                self.profile.disable()
                return self.history
            if self.planner.replan_required:
                # create new plan on updated state
                plan = self.planner.get_plan(self.env)
            else:
                plan = plan[1:]
        if profiling:
            self.profile.disable()
        logging.info("Planning complete")

    def step(self, plan):
        # advance agents
        logging.info(f"plan: {plan}")
        state = list(plan[1])
        for i, s in enumerate(self.env.goal):
            if state[i] == s:
                state[i] = None
        self.env.state = tuple(state)
        self.history.append(self.env.state)
        return self.env.state

    def get_history_as_dataframe(self):
        return convert_history_to_df(self.history)

    def get_history_as_solution(self):
        solution = [[s] for s in self.history[0]]
        for state in self.history[1:]:
            for i, s in enumerate(state):
                if s is not None:
                    solution[i].append(s)
        return solution

    def get_result(self):
        return RunResult(
            self.history,
            create_df_from_profile(self.profile),
            {}
        )


def make_run(scen_path=None, n_agents=2, profiling=None):
    if scen_path is None:
        scen_path = Path() / "benchmark" / "scen-even" / "maze-32-32-4-even-1.scen"
    env = MapfInfoEnvironment(scen_path, n_agents=n_agents)
    planner = CBSPlanner(env, limit=100, discard_conflicts_beyond=3, horizon=3)
    executor = Executor(env, planner)
    executor.run(profiling=profiling)
    print(f"steps in history: {len(executor.history)}")
    return executor


@dataclass
class RunResult:
    history: list
    profile: pd.DataFrame
    config: dict
