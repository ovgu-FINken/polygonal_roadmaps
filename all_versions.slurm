#!/bin/bash

# Time limit is ten minutes (see "man sbatch")
source bin/activate


for N in 2 3 4 5 6 7 8 9 10
do
	for PROBLEM_PARAMS in $(ls benchmark/problem_parameters)
	do
		export SCENARIO="DrivingSwarm;icra2024.yaml;icra2024.yml"
		for PLANNER in $(ls benchmark/planner_config)
		do
			sleep 1
			mkdir -p logs/$PLANNER$SCENARIO/
			sbatch -J "$PLANNER$MAP" --array 1-100 single_job.bash $SCENARIO N $PLANNER $PROBLEM_PARAMS
		done
		sleep 5

		export SCENARIO="DrivingSwarm;icra2024.yaml;icra2024_1m.yml"
		for PLANNER in $(ls benchmark/planner_config)
		do
			sleep 1
			mkdir -p logs/$PLANNER$SCENARIO/
			sbatch -J "$PLANNER$MAP" --array 1-100 single_job.bash $SCENARIO N $PLANNER $PROBLEM_PARAMS
		done
		sleep 5

		export SCENARIO="DrivingSwarm;icra2021_map_no_obstacle.yaml;icra2021.yml"
		for PLANNER in $(ls benchmark/planner_config)
		do
			sleep 1
			mkdir -p logs/$PLANNER$SCENARIO/
			sbatch -J "$PLANNER$MAP" --array 1-100 single_job.bash $SCENARIO N $PLANNER $PROBLEM_PARAMS
		done
		sleep 5
	done
done