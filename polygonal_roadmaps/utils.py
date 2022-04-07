import pandas as pd
import yaml
import pickle
import io
import pstats
import logging
from pathlib import Path
from polygonal_roadmaps import polygonal_roadmap
from polygonal_roadmaps import pathfinding


def read_pickle(planner_config, scen_config, scen):
    try:
        with open(Path("results") / planner_config / scen_config / scen / 'result.pkl', 'rb') as pklfile:
            pkl = pickle.load(pklfile)
        return pkl
    except FileNotFoundError:
        logging.warning(f'File {pklfile} not found')
    return polygonal_roadmap.RunResult([], None, {})


def convert_history_to_df(history):
    records = []
    for t, state in enumerate(history):
        for agent, pos in enumerate(state):
            if pos is None:
                continue
            records.append({'agent': agent, 'x': pos[0], 'y': pos[1], 't': t})
    return pd.DataFrame(records)


def load_results(planner_config, scen_config):
    with open(Path('benchmark') / 'scenario_config' / scen_config) as configfile:
        scen_data = yaml.safe_load(configfile)
    pkls = {}
    for scen in scen_data['scen']:
        pkls[scen] = read_pickle(planner_config, scen_config, scen)
    profile_data = []
    for scen, pkl in pkls.items():
        if pkl.profile is None:
            continue
        df = pkl.profile
        df["scen"] = scen
        env = polygonal_roadmap.MapfInfoEnvironment(scen)
        df['map'] = env.map_file
        df['config'] = pkl.config
        df['planner'] = planner_config
        df['scen'] = scen_config
        profile_data.append(df.loc[df.function.isin(['(astar_path)', '(spacetime_astar)', '(expand_node)', '(step)'])])
    profile_df = pd.concat(profile_data, ignore_index=True)
    return pkls, profile_df


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


def run_all(args):
    if args.loglevel is not None:
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        logging.basicConfig(level=numeric_level)
    for planner in args.planner:
        for scenario in args.scen:
            run_scenarios(scenario, planner, n_agents=args.n_agents, index=args.index)


def create_planner_from_config(config, env):
    if config['planner'] == 'CBS':
        return polygonal_roadmap.CBSPlanner(env, **config['planner_args'])
    elif config['planner'] == 'CCR':
        return polygonal_roadmap.CCRPlanner(env, **config['planner_args'])
    elif config['planner'] == 'PrioritizedPlanner':
        return polygonal_roadmap.PrioritizedPlanner(env, **config['planner_args'])
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
        env = polygonal_roadmap.MapfInfoEnvironment(scen, n_agents=scenario_config['n_agents'])

        planner = create_planner_from_config(planner_config, env)
        path = Path('results') / planner_yml / scenario_yml / scen
        path.mkdir(parents=True, exist_ok=True)
        config = planner_config.update({'map': env.map_file, 'scen': scen})
        run_one(planner, result_path=path / 'result.pkl', config=config)


def run_one(planner, result_path=None, config=None):
    data = None
    try:
        ex = polygonal_roadmap.Executor(planner.env, planner)
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
