import argparse
import yaml
import logging
import pickle

from pathlib import Path

from .polygonal_roadmap import CBSPlanner, Executor, MapfInfoEnvironment
from .polygonal_roadmap import pathfinding


def run_all(args):
    for planner in args.planner:
        for scenario in args.scen:
            run_scenarios(scenario, planner)


def create_planner_from_config(config, env):
    if config['planner'] == 'CBS':
        return CBSPlanner(env, **config['planner_args'])
    raise NotImplementedError(f"planner {config['planner']} does not exist.")


def run_scenarios(scenario_yml, planner_yml):
    with open(Path("benchmark") / 'planner_config' / planner_yml) as stream:
        planner_config = yaml.safe_load(stream)
    with open(Path("benchmark") / 'scenario_config' / scenario_yml) as stream:
        scenario_config = yaml.safe_load(stream)
    for scen in scenario_config['scen']:
        env = MapfInfoEnvironment(scen, n_agents=scenario_config['n_agents'])

        planner = create_planner_from_config(planner_config, env)
        path = Path('results') / planner_yml / scenario_yml
        path.mkdir(parents=True, exist_ok=True)
        run_one(planner, result_path=path / 'result.pkl')


def run_one(planner, result_path=None):
    ex = Executor(planner.env, planner)
    ex.run()
    print(f'n_agents={len(ex.history[0])}')
    print(f'took {len(ex.history)} steps to completion')
    print(f'k-robustness with k={pathfinding.compute_solution_robustness(ex.get_history_as_solution())}')

    logging.info('done')
    if result_path is not None:
        with open(result_path, mode="wb") as results:
            pickle.dump(ex.get_result(), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Pathfinding Experiments")
    parser.add_argument("-planner", type=str, nargs='+', required=True)
    parser.add_argument("-scen", type=str, nargs='+', required=True)
    parser.add_argument("-n_agents", type=int, default=None)
    args = parser.parse_args()
    run_all(args)
