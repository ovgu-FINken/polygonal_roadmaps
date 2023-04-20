from dataclasses import dataclass
from polygonal_roadmaps.environment import Environment, MapfInfoEnvironment
from polygonal_roadmaps.planner import Planner, CBSPlanner
from itertools import groupby
import networkx as nx
from pathlib import Path
from cProfile import Profile
import io
import pstats

import logging

import pandas as pd

class Executor():
    def __init__(self, environment: Environment, planner: Planner, time_frame: int = 1000) -> None:
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
        try:
            plan = self.planner.create_plan(self.env)
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
                    plan = self.planner.create_plan(self.env)
                else:
                    plan = plan[1:]
        except nx.NetworkXNoPath:
            logging.warning("planning failed")
            self.history = []
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
        for i, g in enumerate(self.env.goal):
            solution[i].append(g)
        return solution

    def get_partial_solution(self):
        solution = self.get_history_as_solution()
        # create precedence constraints
        node_visits = {}
        for robot, plan in enumerate(solution):
            for t, node in enumerate(plan):
                if node not in node_visits:
                    node_visits[node] = [(t, robot)]
                else:
                    node_visits[node].append((t, robot))
        for k in node_visits.keys():
            node_visits[k] = [robot for _, robot in sorted(node_visits[k], key=lambda x: x[0])]
            node_visits[k] = [x[0] for x in groupby(node_visits[k])]

        # snip solution
        partial_solution = [[s[0]] for s in solution]
        for robot, plan in enumerate(solution):
            for node in plan[1:]:
                if node_visits[node][0] == robot and partial_solution[robot][-1] != node:
                    partial_solution[robot].append(node)
        return partial_solution

    def get_result(self):
        return RunResult(
            self.history,
            create_df_from_profile(self.profile),
            {},
            planner_step_history=self.planner.get_step_history(),
        )


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
    

def convert_history_to_df(history):
    records = []
    for t, state in enumerate(history):
        for agent, pos in enumerate(state):
            if pos is None:
                continue
            records.append({'agent': agent, 'x': pos[0], 'y': pos[1], 't': t})
    return pd.DataFrame(records)


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
    planner_step_history: list | None = None