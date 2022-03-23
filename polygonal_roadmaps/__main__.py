import argparse
import yaml
import logging

from .polygonal_roadmap import CBSPlanner, MapfInfoEnvironment


def main(config):
    env = MapfInfoEnvironment(config['env']['scen'], n_agents=config['env']['n_agents'])
    if config['planner'] == 'CBS':
        print(config)
    else:
        logging.warn(f"Planner '{config['planner']}' not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Pathfinding Experiments")
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-scen", type=str, default=None)
    parser.add_argument("-n_agents", type=int, default=None)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if args.scen is not None:
        config["env"]["scen"] = args.scen
    if args.n_agents is not None:
        config["env"]["n_agents"] = args.n_agents
    main(config)
