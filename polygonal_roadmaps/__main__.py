import argparse

from .polygonal_roadmap import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Pathfinding Experiments")
    parser.add_argument("-planner", type=str, nargs='+', required=True)
    parser.add_argument("-scentype", type=str, default="even")
    parser.add_argument("-maps", type=str, nargs='+', required=True)
    parser.add_argument("-n_agents", type=int, default=10)
    parser.add_argument("-index", type=int, default=None)
    parser.add_argument("-timelimit", type=int, default=None, help="Memory Limit in Gb")
    parser.add_argument("-memlimit", type=int, default=None, help="Memory Limit in Gb")
    parser.add_argument("-loglevel", type=str, default=None)
    parser.add_argument("-logfile", type=str, default=None)
    args = parser.parse_args()
    utils.run_all(args)
