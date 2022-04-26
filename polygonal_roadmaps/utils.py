import pandas as pd
import yaml
import pickle
import io
import pstats
import logging
import glob
import resource
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
        df = {}
        df["scen"] = cfg[2]
        df["scentype"] = cfg[1]
        # env = polygonal_roadmap.MapfInfoEnvironment(Path(even) / scen[1])
        # df['config'] = pkl.config
        df['planner_file'] = cfg[0]
        df['k'] = pkl.k
        # df['SOC'] = pkl.sum_of_cost
        df['sum_of_cost'] = pkl.soc
        df['failed'] = pkl.failed or pkl.k <= 0
        df['makespan'] = pkl.makespan
        d = pkl.profile
        df['spatial_astar'] = d.loc[d.function.eq('(astar_path)') | d.function.eq('(nx_shortest)'), "ncalls"].astype(int).sum()
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
        logging.info(f"set memlimit to {args.limit}min == {limit}s")
        resource.setrlimit(resource.RLIMIT_CPU, (limit, hardlimit))
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
        planner_config.update({'map_file': env.map_file, 'scen': scen})
        run_one(planner, result_path=path / 'result.pkl', config=planner_config)


def run_one(planner, result_path=None, config=None):
    data = None
    ex = polygonal_roadmap.Executor(planner.env, planner)
    ex.failed = True
    try:
        print('-----------------')
        if result_path is not None:
            print(f'{result_path}')
        ex.run()
        ex.failed = False
    except Exception as e:
        ex.profile.disable()
        data.k = -1
        data.makespan = -1
        data.soc = -1
        logging.warning(f'Exception occured during execution:\n{e}')
        raise e
    finally:
        data = ex.get_result()
        data.failed = ex.failed
        if len(ex.history):
            data.soc = pathfinding.sum_of_cost(ex.get_history_as_solution(), graph=ex.env.g, weight="dist")
            data.makespan = len(ex.history)
            data.k = pathfinding.compute_solution_robustness(ex.get_history_as_solution())
        else:
            data.soc = 0
            data.makespan = 0
            data.k = -1
        print(f'took {len(ex.history)} steps to completion')
        data.config = config
        if result_path is not None:
            with open(result_path, mode="wb") as results:
                pickle.dump(data, results)
        print(f'k-robustness with k={data.k}')
        print("failed" if data.failed else "succeeded")
        print('-----------------')
