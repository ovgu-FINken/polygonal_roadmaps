import pandas as pd
import yaml
import pickle
from pathlib import Path
from polygonal_roadmaps import polygonal_roadmap


def read_pickle(config, scen):
    with open(Path("results") / config / scen / 'result.pkl', 'rb') as pklfile:
        pkl = pickle.load(pklfile)
    return pkl


def load_results(config):
    with open(Path('benchmark') / 'config' / config) as configfile:
        config_data = yaml.safe_load(configfile)
    pkls = {}
    for scen in config_data['env']['scen']:
        pkls[scen] = read_pickle(config, scen)
    profile_data = []
    for scen, pkl in pkls.items():
        df = pkl.profile
        df["scen"] = scen
        env = polygonal_roadmap.MapfInfoEnvironment(scen)
        df['map'] = env.map_file
        df['config'] = config
        profile_data.append(df.loc[df.function.isin(['(astar_path)', '(spacetime_astar)'])])
    profile_df = pd.concat(profile_data, ignore_index=True)
    return config_data, pkls, profile_df
