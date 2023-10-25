#!/bin/bash

# Time limit is ten minutes (see "man sbatch")
source bin/activate
export SCENARIO="DrivingSwarm;icra2024.yaml;icra2024_1m.yml"
export N_AGENTS=3

for PLANNER in $(ls benchmark/planner_config)
do
	python3 -m polygonal_roadmaps -index 0 -n_runs 5 -n_agents $N_AGENTS -planner $PLANNER -scenario $SCENARIO
done
