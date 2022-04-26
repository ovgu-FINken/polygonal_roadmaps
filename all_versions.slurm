#!/bin/bash

# Time limit is ten minutes (see "man sbatch")
source bin/activate
for PLANNER in $(ls benchmark/planner_config)
do
	for MAP in $(ls benchmark/maps/ | head -n $1)
	do
		sleep 1
		mkdir -p logs/$PLANNER$MAP/
		sbatch -J "$PLANNER$MAP" --array 1-25 single_job.bash $PLANNER $MAP
	done
done
