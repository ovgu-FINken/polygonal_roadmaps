from dataclasses import dataclass
from polygonal_roadmaps.environment import Environment, MapfInfoEnvironment
from polygonal_roadmaps.planning import Planner, CBSPlanner, sum_of_cost, compute_solution_robustness, Plans
from itertools import groupby
import networkx as nx
from pathlib import Path
from cProfile import Profile
import io
import pstats
from icecream import ic

import logging

import pandas as pd
from random import shuffle

def replace_goal_with_none(state, goal):
    return tuple([None if s == g else s for s, g in zip(state, goal)])

def next_state_is_valid(next_state, env):
    if len(next_state) != len(env.state):
        return False
    # we shoud not get a plan without any progress though this is technically possible
    if next_state == env.state:
        return False
    for s, ns in zip(env.state, next_state):
        if s is None:
            assert ns is None
        if ns is None:
            continue
        if ns == s:
            continue
        if not env.g.has_edge(s, ns):
            return False
    nodes = [n for n in next_state if n is not None]
    if len(nodes) != len(set(nodes)):
        return False
    
    return True

def advance_state_randomly(env, next_state):
    assert next_state_is_valid(next_state, env), f"transition{env.state} -> {next_state} is not valid"
    if not env.planning_problem_parameters.step_num:
        return next_state
    step_num = env.planning_problem_parameters.step_num
    state = env.state
    if step_num >= len(state):
        return next_state
    # the bitmask wil be [True, True, True .... False, False, False]
    bitmask = [True] * step_num + [False] * (len(state) - step_num)
    
    # we randomize the oreder of the bitmask
    shuffle(bitmask)
    ret = []
    for s, ns, b in zip(state, next_state, bitmask):
        if b:
            ret.append(ns)
        else:
            ret.append(s)

    # retry if we did not make any progress, i.e. the state we progress is a wait action
    # replanning should remove wait actions by the state change
    if ret == state:
        return advance_state_randomly(env, next_state)
    return ret

class Executor():
    def __init__(self, environment: Environment, planner: Planner, time_frame: int = 1000) -> None:
        self.env = environment
        self.env.state = self.env.start
        self.planner = planner
        self.history = []
        self.plans = []
        self.time_frame = time_frame

    def run(self, profiling=False, replan=None):
        if replan is not None:
            self.replan = replan
        else:
            self.replan = self.planner.replan_required
        if self.env.planning_problem_parameters.step_num: 
            if self.env.planning_problem_parameters.step_num < len(self.env.state):
                self.replan = True
        self.failed = True
        if len(self.history) >= self.time_frame:
            return
        if profiling:
            self.profile = Profile()
            self.profile.enable()
        try:
            plan = self.planner.create_plan(self.env)
            if not plan:
                return self.history
            for i in range(self.time_frame):
                logging.info(f"At iteration {i} / {self.time_frame}")
                self.step(plan)
                # (goal is reached)
                if all(s is None for s in self.env.state):
                    # plan = plan[1:]
                    if profiling:
                        self.profile.disable()
                    if Plans(self.history).is_valid(self.env):
                        self.failed = False
                    return self.history
                if self.replan:
                    # create new plan on updated state
                    plan = self.planner.create_plan(self.env)
                else:
                    plan = Plans([p[1:] for p in plan])
        except nx.NetworkXNoPath:
            logging.warning("planning failed")
            self.failed = True
        if profiling:
            self.profile.disable()
        logging.info("Planning complete")

    def step(self, plan):
        # advance agents
        logging.info(f"plan: {plan}")
        self.history.append(self.env.state)
        self.plans.append(plan)
        state = advance_state_randomly(self.env, plan.get_next_state())
        assert next_state_is_valid(state, self.env), f"transition{self.env.state} -> {state} is not valid"
        self.env.state = replace_goal_with_none(state, self.env.goal)
        return self.env.state

    def get_history_as_dataframe(self):
        # self.history is a list of states, each state is a tuple of agent positions
        # we want to create a dataframe where each row is a timestep and each column is an agent
        # in addition, we want to get the agents position in the environment
        # and the plan that the agent is executing
        records = []
        for t, states in enumerate(self.history):
            for i, state in enumerate(states):
                records.append({
                    't': t,
                    'agent': i,
                    'state': state,
                })
                position = self.env.get_position(state)
                if position is not None:
                    records[-1]['x'] = position[0]
                    records[-1]['y'] = position[1]
        return pd.DataFrame(records)

    def get_history_as_solution(self):
        # create plans object form history
        p = Plans.from_state_list(self.history)
        return p.plans

    def get_result(self):
        return RunResult(
            self.history,
            create_df_from_profile(self.profile),
            {},
            None # planner_step_history=self.planner.get_step_history(),
        )
    
    def run_results(self):
        # return the aggregated results of a run, i.e. flowtime, makespan, etc. as well as the cost of the run
        print("compile run results")
        solution = self.get_history_as_solution()
        result = {}
        result["steps"] = len(self.history)
        result["flowtime"] = sum([len(x) for x in self.history]) / len(self.env.state)
        #result["makespan"] = max([ sum_of_cost([solution[i]]) for i, _ in enumerate(self.env.goal)])
        result["sum_of_cost"] = sum_of_cost(solution, graph=self.env.g)
        result["failed"] = self.failed
        #result["robustness"] = compute_solution_robustness(solution)
        result["astar"] = 0
        result["spacetime_astar"] = 0
        return result
        
    def run_history(self):
        # return the history of a run, i.e. the state of the agents at each timestep, intermediate plans, etc.
        # everything needed to make a video or otherwise visualize the run
        print("compile run history")
        return self.get_history_as_dataframe()


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