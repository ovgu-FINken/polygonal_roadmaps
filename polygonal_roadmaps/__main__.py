import argparse
import yaml
import logging
import pickle

from pathlib import Path

from .polygonal_roadmap import CBSPlanner, CCRPlanner, Executor, MapfInfoEnvironment
from .polygonal_roadmap import pathfinding


def run_all(args):
    if args.loglevel is not None:
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        logging.basicConfig(level=numeric_level)
    for planner in args.planner:
        for scenario in args.scen:
            run_scenarios(scenario, planner, n_agents=args.n_agents, index=args.index)


def create_planner_from_config(config, env):
    if config['planner'] == 'CBS':
        return CBSPlanner(env, **config['planner_args'])
    elif config['planner'] == 'CCR':
        return CCRPlanner(env, **config['planner_args'])
    raise NotImplementedError(f"planner {config['planner']} does not exist.")


def run_scenarios(scenario_yml, planner_yml, n_agents=None, index=None):
    with open(Path("benchmark") / 'planner_config' / planner_yml) as stream:
        planner_config = yaml.safe_load(stream)
    with open(Path("benchmark") / 'scenario_config' / scenario_yml) as stream:
        scenario_config = yaml.safe_load(stream)
        if n_agents is not None:
            scenario_config['n_agents'] = n_agents
    if index is None:
        scenarios = [scen for scen in scenario_config['scen']]
    else:
        scenarios = [scenario_config['scen'][index]]
    for scen in scenarios:
        env = MapfInfoEnvironment(scen, n_agents=scenario_config['n_agents'])

        planner = create_planner_from_config(planner_config, env)
        path = Path('results') / planner_yml / scenario_yml / scen
        path.mkdir(parents=True, exist_ok=True)
        config = planner_config.update({'map': env.map_file, 'scen': scen})
        run_one(planner, result_path=path / 'result.pkl', config=config)


def run_one(planner, result_path=None, config=None):
    data = None
    try:
        ex = Executor(planner.env, planner)
        print('-----------------')
        if result_path is not None:
            print(f'{result_path}')
        ex.run()
        print(f'n_agents={len(ex.history[0])}')
        print(f'took {len(ex.history)} steps to completion')
        print(f'k-robustness with k={pathfinding.compute_solution_robustness(ex.get_history_as_solution())}')
        print('-----------------')

        data = ex.get_result()
        data.config = config
        logging.info('done')
    except Exception as e:
        if result_path is not None:
            with open(result_path, mode="wb") as results:
                pickle.dump(data, results)
        logging.warning(f'Exception occured during execution:\n{e}')
        raise e
    finally:
        if result_path is not None:
            with open(result_path, mode="wb") as results:
                pickle.dump(data, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Pathfinding Experiments")
    parser.add_argument("-planner", type=str, nargs='+', required=True)
    parser.add_argument("-scen", type=str, nargs='+', required=True)
    parser.add_argument("-n_agents", type=int, default=None)
    parser.add_argument("-index", type=int, default=None)
    parser.add_argument("-loglevel", type=str, default=None)
    args = parser.parse_args()
    run_all(args)
