import pandas as pd
import yaml
import pickle
import io
import pstats
import logging
import glob
from tqdm import tqdm
from pathlib import Path
from polygonal_roadmaps import polygonal_roadmap
from polygonal_roadmaps import pathfinding


def read_pickle(location):
    try:
        with open(location, 'rb') as pklfile:
            pkl = pickle.load(pklfile)
        return pkl
    except FileNotFoundError:
        logging.warning(f'File {location} not found')
    return polygonal_roadmap.RunResult([], None, {})


def convert_history_to_df(history):
    records = []
    for t, state in enumerate(history):
        for agent, pos in enumerate(state):
            if pos is None:
                continue
            records.append({'agent': agent, 'x': pos[0], 'y': pos[1], 't': t})
    return pd.DataFrame(records)


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
        df = pkl.profile
        df["scen"] = cfg[2]
        df["scentype"] = cfg[1]
        # env = polygonal_roadmap.MapfInfoEnvironment(Path(even) / scen[1])
        df['map_file'] = cfg[2].split(cfg[1])[0][:-1]
        # df['config'] = pkl.config
        df['planner'] = cfg[0]
        df['robustness'] = pkl.k
        if hasattr(pkl, 'makespan'):
            df['makespan'] = pkl.makespan
        # df['SOC'] = pkl.sum_of_cost
        profile_data.append(df.loc[df.function.isin(['(astar_path)', '(spacetime_astar)', '(run)', '(nx_shortest)'])])
    if profile_data:
        profile_df = pd.concat(profile_data, ignore_index=True)
    else:
        logging.warning("something went wrong, profile data is empty(?)")
        profile_df = pd.DataFrame()
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
        logging.basicConfig(level=numeric_level, filename=args.logfile)
    for planner in args.planner:
        for map_file in args.maps:
            run_scenarios(map_file.split(".")[0], planner, n_agents=args.n_agents, index=args.index, scentype=args.scentype)


def create_planner_from_config(config, env):
    if config['planner'] == 'CBS':
        return polygonal_roadmap.CBSPlanner(env, **config['planner_args'])
    elif config['planner'] == 'CCR':
        return polygonal_roadmap.CCRPlanner(env, **config['planner_args'])
    elif config['planner'] == 'PrioritizedPlanner':
        return polygonal_roadmap.PrioritizedPlanner(env, **config['planner_args'])
    raise NotImplementedError(f"planner {config['planner']} does not exist.")


def run_scenarios(map_file, planner_yml, n_agents=10, index=None, scentype="even"):
    with open(Path("benchmark") / 'planner_config' / planner_yml) as stream:
        planner_config = yaml.safe_load(stream)
    if index is None:
        scenarios = glob.glob(f"benchmark/scen/{scentype}/{map_file}-{scentype}-*.scen")
        # strip path from scenario
        scenarios = ['{scentype}/' + s.split('/')[-1] for s in scenarios]
    else:
        scenarios = [f"{scentype}/{map_file}-{scentype}-{index}.scen"]
    for scen in scenarios:
        env = polygonal_roadmap.MapfInfoEnvironment(scen, n_agents=n_agents)

        planner = create_planner_from_config(planner_config, env)
        path = Path('results') / planner_yml / scen
        path.mkdir(parents=True, exist_ok=True)
        config = planner_config.update({'map': env.map_file, 'scen': scen})
        run_one(planner, result_path=path / 'result.pkl', config=config)


def run_one(planner, result_path=None, config=None):
    data = None
    ex = polygonal_roadmap.Executor(planner.env, planner)
    ex.failed = False
    try:
        print('-----------------')
        if result_path is not None:
            print(f'{result_path}')
        ex.run()
    except Exception as e:
        ex.profile.disable()
        ex.failed = True
        logging.warning(f'Exception occured during execution:\n{e}')
        raise e
    finally:
        data = ex.get_result()
        print(f'n_agents={len(ex.env.start)}')
        if ex.history:
            print(f'took {len(ex.history)} steps to completion')
            if not ex.failed:
                data.k = pathfinding.compute_solution_robustness(ex.get_history_as_solution())
            else:
                data.k = -1
        else:
            data.k = -1
        data.config = config
        data.failed = ex.failed
        data.makespan = len(ex.history)
        if result_path is not None:
            with open(result_path, mode="wb") as results:
                pickle.dump(data, results)
        print(f'k-robustness with k={data.k}')
        print('-----------------')
