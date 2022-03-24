import argparse
import yaml
import logging
import pickle

from pathlib import Path

from .polygonal_roadmap import CBSPlanner, Executor, MapfInfoEnvironment
from .polygonal_roadmap import pathfinding


def run_all(config):
    assert config['planner'] in ['CBS']
    for scen in config['env']['scen']:
        env = MapfInfoEnvironment(scen, n_agents=config['env']['n_agents'])
        run_one(config, env)


def run_one(config, env):
    if config['planner'] == 'CBS':
        planner = CBSPlanner(env, **config['planner_args'])
    else:
        logging.warn(f"Planner '{config['planner']}' not found")
        return
    ex = Executor(env, planner)
    ex.run()
    print(f'n_agents={len(ex.history[0])}')
    print(f'took {len(ex.history)} steps to completion')
    print(f'k-robustness with k={pathfinding.compute_solution_robustness(ex.get_history_as_solution())}')

    logging.info('done')
    if config['results_location'] is not None:
        path = Path(f'{config["results_location"]}') / config['config_file'] / env.scenario_file
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'result.pkl', mode="wb") as results:
            pickle.dump(ex.get_result(), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Pathfinding Experiments")
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-scen", type=str, nargs='+', default=None)
    parser.add_argument("-n_agents", type=int, default=None)
    parser.add_argument("-results_location", type=str, default=None)
    args = parser.parse_args()
    with open(Path("benchmark") / 'config' / args.config) as stream:
        config = yaml.safe_load(stream)
    if args.scen is not None:
        config["env"]["scen"] = args.scen
    if args.n_agents is not None:
        config["env"]["n_agents"] = args.n_agents
    config["results_location"] = args.results_location
    if config["results_location"] is None:
        config["results_location"] = "results/"
    config['config_file'] = args.config
    run_all(config)
