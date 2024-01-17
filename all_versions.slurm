#!/bin/bash

# Time limit is ten minutes (see "man sbatch")
source bin/activate

for PROBLEM_PARAMS in $(ls benchmark/problem_params)
	do
	export SCENARIO="DrivingSwarm;icra2024.yaml;icra2024.yml"
	for PLANNER in $(ls benchmark/planner_config)
	do
		sleep 1
		mkdir -p logs/$PLANNER$SCENARIO/
		sbatch -J "$PLANNER$MAP" --array 1-31 single_job.bash $SCENARIO 5 $PLANNER $PROBLEM_PARAMS
	done
	sleep 5

	export SCENARIO="DrivingSwarm;icra2024.yaml;icra2024_1m.yml"
	for PLANNER in $(ls benchmark/planner_config)
	do
		sleep 1
		mkdir -p logs/$PLANNER$SCENARIO/
		sbatch -J "$PLANNER$MAP" --array 1-31 single_job.bash $SCENARIO 5 $PLANNER $PROBLEM_PARAMS
	done
	sleep 5
done
