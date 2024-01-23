from dataclasses import dataclass
from polygonal_roadmaps.environment import Environment, MapfInfoEnvironment
from polygonal_roadmaps.planning import Planner, CBSPlanner, sum_of_cost, Plans
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
import numpy as np

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
            if ns is not None:
                if ns in env.goal:
                    logging.warn("next state is a goal position, likely the planner does not use the current state as start state")
                    continue
                logging.warn("next state is not valid, agent is not allowed to move from None to a position")
                return False
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

def advance_state_randomly(env: Environment, next_state):
    assert next_state_is_valid(next_state, env), f"transition{env.state} -> {next_state} is not valid, goal state is {env.goal}"
    step_num = env.planning_problem_parameters.step_num
    next_state = replace_goal_with_none(next_state, env.goal)
    if not step_num or step_num >= len(env.state):
        return next_state

    
    state = list(env.state)
    for _ in range(step_num):
        # get all agents that are not at their goal
        agents = [i for i, s in enumerate(state) if s is not None and s != env.goal[i] and s != next_state[i]]
        progress_state = np.random.choice(agents)
        state[progress_state] = next_state[progress_state]
        state = replace_goal_with_none(state, env.goal)
        if state == next_state:
            break
        if state == env.goal:
            break

    return state

class Executor():
    def __init__(self, environment: Environment, planner: Planner, time_frame: int = 1000) -> None:
        self.env = environment
        self.env.state = self.env.start
        self.planner = planner
        self.history = []
        self.plans = []
        self.time_frame = time_frame
        self.finished = False

    def run(self, profiling=False, replan=None):
        print("executor run")
        if replan is not None:
            self.replan = replan
        else:
            self.replan = self.planner.replan_required

        if self.env.planning_problem_parameters.step_num: 
            if self.env.planning_problem_parameters.step_num < len(self.env.state):
                self.replan = True
        self.failed = True
        if len(self.history) >= self.time_frame:
            logging.warn("return history as time frame is reached, should not happen for fresh run")
            return
        if profiling:
            self.profile = Profile()
            self.profile.enable()
        try:
            plan = self.planner.create_plan(self.env)
            if not plan:
                logging.warn("no plan, return history")
                return self.history
            for i in range(self.time_frame):
                logging.info(f"At iteration {i} / {self.time_frame}")
                self.step(plan)
                # (goal is reached)
                if all(s is None for s in self.env.state):
                    # plan = plan[1:]
                    if profiling:
                        self.profile.disable()
                    self.failed = not Plans(self.get_history_as_solution()).is_valid(self.env)
                    if self.failed:
                        logging.warn(f"plans in history are not valid: {self.history}")
                    self.finished = True
                    return self.history
                if self.replan:
                    # create new plan on updated state
                    plan = self.planner.create_plan(self.env)
                else:
                    plan = Plans([p[1:] for p in plan])
                logging.warning(f"plan: {plan}")
                logging.warning(f"state: {self.env.state}")
                logging.warning(f"goal: {self.env.goal}")
                logging.warning(f"goal_transition_valid: {next_state_is_valid(self.env.goal, self.env)}")
        except nx.NetworkXNoPath:
            logging.warning("planning failed")
            self.failed = True
        if profiling:
            self.profile.disable()
        logging.info("Planning complete")
        return self.history

    def step(self, plan: Plans):
        # advance agents
        logging.info(f"plan: {plan}")
        # plan only has to be valid up until planning horizon, we will check validity of the transition anyway
        #assert plan.is_valid(self.env), f"plan {plan} is not valid"
        self.history.append(self.env.state)
        self.plans.append(plan)
        state = advance_state_randomly(self.env, plan.get_next_state())
        assert next_state_is_valid(state, self.env), f"transition{self.env.state} -> {state} is not valid, goal state is {self.env.goal}"
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
        logging.info("compile run results")
        solution = self.get_history_as_solution()
        result = {}
        result["steps"] = len(self.history)
        result["flowtime"] = sum([len(x) for x in self.history]) / len(self.env.state)
        #result["makespan"] = max([ sum_of_cost([solution[i]]) for i, _ in enumerate(self.env.goal)])
        result["sum_of_cost"] = sum_of_cost(solution, graph=self.env.g)
        result["finished"] = self.finished
        result["valid"] = not self.failed
        #result["robustness"] = compute_solution_robustness(solution)
        result["astar"] = 0
        result["spacetime_astar"] = 0
        return result
        
    def run_history(self):
        # return the history of a run, i.e. the state of the agents at each timestep, intermediate plans, etc.
        # everything needed to make a video or otherwise visualize the run
        logging.info("compile run history")
        logging.info(f"len history: {len(self.history)}")
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