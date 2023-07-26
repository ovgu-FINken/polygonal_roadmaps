import argparse
from typing import Iterable
import pandas as pd
import signal
import yaml
import pickle
import logging
import glob
import resource
from tqdm import tqdm
from pathlib import Path
from polygonal_roadmaps import geometry, planning, polygonal_roadmap
from polygonal_roadmaps.environment import PlanningProblemParameters, Environment, RoadmapEnvironment, MapfInfoEnvironment

class TimeoutError(Exception):
    pass


def read_pickle(location):
    try:
        with open(location, 'rb') as pklfile:
            pkl = pickle.load(pklfile)
        return pkl
    except FileNotFoundError:
        logging.warning(f'File {location} not found')
    except pickle.UnpicklingError:
        logging.warning(f'could not upnpickle {location}')
    except MemoryError:
        logging.warning(f'MemoryError while unpickle {location}')
    except EOFError:
        logging.warning(f'EOF while unpickle {location}')
    except TypeError:
        logging.warning(f'Type Error while unpickle {location}')
    return polygonal_roadmap.RunResult([], None, {})


def load_results(path=None):
    if path is None:
        path = "results"
    result_files = glob.glob(f"{path}/*/*/**/result.pkl")
    logging.debug(f'loading results: {result_files}')
    pkls = {}
    for dings in result_files:
        _, planner_config, even, scen, *_ = dings.split('/')
        pkls[planner_config, even, scen] = read_pickle(dings)
    profile_data = []
    for cfg, pkl in tqdm(pkls.items()):
        if pkl is None:
            logging.warning(f"skipping {cfg}, pkl is None")
            continue
        if pkl.profile is None:
            logging.warning(f"skipping {cfg}, profile is None")
            continue
        df = {}
        df["scen"] = cfg[2]
        df["scentype"] = cfg[1]
        # env = polygonal_roadmap.MapfInfoEnvironment(Path(even) / scen[1])
        # df['config'] = pkl.config
        df['planner_file'] = cfg[0]
        df['k'] = pkl.k
        # df['SOC'] = pkl.sum_of_cost
        df['sum_of_cost'] = pkl.soc
        df['failed'] = pkl.failed
        df['makespan'] = pkl.makespan
        d = pkl.profile
        df['spatial_astar'] = d.loc[d.function.eq('(astar_path)') |
                                    d.function.eq('(shortest_path_length)'), "ncalls"].astype(int).sum()
        df['spacetime_astar'] = d.loc[d.function.eq('(spacetime_astar)'), "ncalls"].astype(int).sum()
        if pkl.config is None:
            logging.warn(f'config is None in pkl {cfg[0]}, {cfg[2]}')
            logging.warn(f'{pkl}')
        else:
            df['scen'] = pkl.config['scen']
            df['map_file'] = pkl.config['map_file']
            df['planner'] = pkl.config['planner']
            for k, v in pkl.config['planner_args'].items():
                df[k] = v

        profile_data.append(df)
    if profile_data:
        profile_df = pd.DataFrame(profile_data)
    else:
        logging.warning("something went wrong, profile data is empty(?)")
        profile_df = pd.DataFrame()
    return pkls, profile_df


def raise_timeout(number, _):
    logging.warning(f"raising exception because of signal {number}")
    raise TimeoutError()


def run_all(args):
    # run all jobs specified by args

    # set signal handelrs for sigterm, sigxcpu
    # this is needed to return on cluster signals and for setting CPU-time limit
    signal.signal(signal.SIGXCPU, raise_timeout)
    signal.signal(signal.SIGTERM, raise_timeout)

    if args.loglevel is not None:
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        logging.basicConfig(level=numeric_level, filename=args.logfile)
    if args.memlimit is not None:
        # set memlimit, arg is in GB
        softlimit, hardlimit = resource.getrlimit(resource.RLIMIT_AS)
        logging.info(f"sl: {softlimit}, hl: {hardlimit}")
        limit = args.memlimit << (10 * 3)
        logging.info(f"set memlimit to {args.memlimit}Gb == {limit}b")
        resource.setrlimit(resource.RLIMIT_AS, (limit, hardlimit))
    if args.timelimit is not None:
        # set memlimit, arg is in GB
        softlimit, hardlimit = resource.getrlimit(resource.RLIMIT_CPU)
        logging.info(f"sl: {softlimit}, hl: {hardlimit}")
        limit = args.timelimit * 60
        logging.info(f"set memlimit to {args.timelimit}min == {limit}s")
        resource.setrlimit(resource.RLIMIT_CPU, (limit, hardlimit))
    for planner in args.planner:
        for scenario in args.scenarios:
            run_scenario(scenario, planner, n_agents=args.n_agents, index=args.index, problem_parameters=args.problem_parameters)


def create_planner_from_config_file(config_file, env):
    with open(config_file) as stream:
        planner_config = yaml.safe_load(stream)
    return create_planner_from_config(planner_config, env)


def create_planner_from_config(config, env) -> planning.Planner:
    if config['planner'] == 'CBS':
        return planning.CBSPlanner(env, **config['planner_args'])
    elif config['planner'] == 'CCR':
        return planning.CCRPlanner(env, **config['planner_args'])
    elif config['planner'] == 'CCRv2':
        return planning.CCRv2(env, **config['planner_args'])
    elif config['planner'] == 'PrioritizedPlanner':
        return planning.PrioritizedPlanner(env, **config['planner_args'])
    raise NotImplementedError(f"planner {config['planner']} does not exist.")


def run_scenario(scen_str:str, planner_config_file:str, n_agents:int=10, index:None|int=None, n_scenarios:int=25, problem_parameters:str|None=None):
    planner_file = Path("benchmark") / 'planner_config' / planner_config_file
    with open(planner_file) as stream:
        planner_config = yaml.safe_load(stream)
    data = []
    for i, env in enumerate(env_generator(scen_str, n_agents, index=index, problem_parameters=problem_parameters)):
        if i>=n_scenarios:
            break
        planner = create_planner_from_config(planner_config, env)
        path = Path('results') / planner_config_file / scen_str
        path.mkdir(parents=True, exist_ok=True)
        planner_config.update({'scen': scen_str, "index": i})
        data.append(run_one(planner, result_path=path / 'result.pkl', config=planner_config))
    return data


def load_problem_parameters(problem_parameters:str|None):
    if problem_parameters is None:
        return PlanningProblemParameters()
    with open(Path("benchmark") / 'problem_parameters' / problem_parameters) as stream:
        return PlanningProblemParameters(**yaml.safe_load(stream))


def env_generator(scen_str, n_agents=10, index=None, problem_parameters:str|None=None) -> Iterable[Environment]:
    problem_parameters = load_problem_parameters(problem_parameters)
    envs: list[Environment] = []
    match scen_str.split(";"):
        case "MAPF", map_file, scen_type:
            envs = load_mapf_scenarios(map_file, scen_type, n_agents, index=index, planning_problem_parameters=problem_parameters)
        case "DrivingSwarm", map_yml, scenario_yml:
            envs = load_driving_swarm_scenarios(map_yml, scenario_yml, n_agents, index=index, planning_problem_parameters=problem_parameters)
    return envs


def load_driving_swarm_scenarios(map_yml, scenario_yml, n_agents, index=None, planning_problem_parameters=PlanningProblemParameters()) -> Iterable[Environment]:
    scenario = None
    with open(Path("benchmark") / "DrivingSwarm" / "scenarios" / scenario_yml) as stream:
        scenario = yaml.safe_load(stream)
    g = scenario["generator"]
    generators = None
    if g["type"] == "square":
        generators = geometry.square_tiling(g["grid_size"], working_area_x=g["wx"], working_area_y=g["wy"])
    if g["type"] == "hex":
        generators = geometry.hexagon_tiling(g["grid_size"], working_area_x=g["wx"], working_area_y=g["wy"])
    env = RoadmapEnvironment(Path("benchmark") / "DrivingSwarm" / "maps" / map_yml, 
                             scenario["start"], scenario["goal"], 
                             planning_problem_parameters=planning_problem_parameters, 
                             generator_points=generators, offset=g["offset"],
                             wx=g["wx"], wy=g["wy"])
    return [env]


def load_mapf_scenarios(map_file, scentype, n_agents, index=None, planning_problem_parameters=PlanningProblemParameters()) -> Iterable[Environment]:
    if index is None:
        scenarios = glob.glob(f"benchmark/scen/{scentype}/{map_file}-{scentype}-*.scen")
        # strip path from scenario
        scenarios = [f'{scentype}/' + s.split('/')[-1] for s in scenarios]
    else:
        scenarios = [f"{scentype}/{map_file}-{scentype}-{index}.scen"]
    return (MapfInfoEnvironment(scen, n_agents=n_agents, planning_problem_parameters=planning_problem_parameters) for scen in scenarios)
    

def run_one(planner, result_path=None, config=None):
    data = None
    ex = polygonal_roadmap.Executor(planner.environment, planner)
    ex.failed = True
    try:
        print('-----------------')
        print(ex.env)
        if result_path is not None:
            print(f'{result_path}')
        ex.run()
        ex.failed = False
    except (MemoryError, TimeoutError):
        print("out of computational resources")
        _, hardlimit = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (hardlimit, hardlimit))
        _, hardlimit = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (hardlimit, hardlimit))
        ex.profile.disable()
    except Exception as e:
        ex.profile.disable()
        logging.warning(f'Exception occured during execution:\n{e}')
        raise e
    finally:
        # reset resource limit before saving results (we don't want to trigger this during result)
        data = ex.get_result()
        data.failed = ex.failed
        if not ex.failed and len(ex.history):
            data.soc = planning.sum_of_cost(ex.get_history_as_solution(), graph=ex.env.g, weight="dist")
            data.makespan = len(ex.history)
            data.k = planning.compute_solution_robustness(ex.get_history_as_solution())
            data.steps = sum([len(p) for p in ex.get_history_as_solution()])
        else:
            data.soc = -1
            data.makespan = -1
            data.k = -1
            data.steps = -1
        print(f'took {len(ex.history)} steps to completion')
        data.config = config
        if result_path is not None:
            with open(result_path, mode="wb") as results:
                pickle.dump(data, results)
        print(f'k-robustness with k={data.k}')
        print("failed" if data.failed else "succeeded")
        print('-----------------')
    return data


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Runner for Pathfinding Experiments")
    parser.add_argument("-planner", type=str, nargs='+', required=True)
    parser.add_argument("-scenarios", type=str, nargs='+', required=True)
    parser.add_argument("-problem_parameters", type=str, default=None)
    parser.add_argument("-n_agents", type=int, default=10)
    parser.add_argument("-index", type=int, default=None)
    parser.add_argument("-timelimit", type=int, default=None, help="Memory Limit in Gb")
    parser.add_argument("-memlimit", type=int, default=None, help="Memory Limit in Gb")
    parser.add_argument("-loglevel", type=str, default=None)
    parser.add_argument("-logfile", type=str, default=None)
    args = parser.parse_args()
    run_all(args)
    

if __name__ == '__main__':
    cli_main()
        