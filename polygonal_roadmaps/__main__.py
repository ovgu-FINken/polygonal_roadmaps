import argparse
import yaml
import logging
import pickle

from .polygonal_roadmap import CBSPlanner, Executor, MapfInfoEnvironment
from .polygonal_roadmap import pathfinding


def main(config):
    env = MapfInfoEnvironment(config['env']['scen'], n_agents=config['env']['n_agents'])
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
    if config['results_file'] is not None:
        with open(config['results_file'], mode="wb") as results:
            pickle.dump(ex.get_result(), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Pathfinding Experiments")
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-scen", type=str, default=None)
    parser.add_argument("-n_agents", type=int, default=None)
    parser.add_argument("-store_result", type=str, default=None)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if args.scen is not None:
        config["env"]["scen"] = args.scen
    if args.n_agents is not None:
        config["env"]["n_agents"] = args.n_agents
    config["results_file"] = args.store_result
    main(config)
