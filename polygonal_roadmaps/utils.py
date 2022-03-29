import pandas as pd
import yaml
import pickle
import io
import pstats
from pathlib import Path
from polygonal_roadmaps import polygonal_roadmap


def read_pickle(planner_config, scen_config, scen):
    with open(Path("results") / planner_config / scen_config / scen / 'result.pkl', 'rb') as pklfile:
        pkl = pickle.load(pklfile)
    return pkl


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
