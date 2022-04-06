#!/bin/bash

# Time limit is ten minutes (see "man sbatch")
#SBATCH --time=24:00:00
# Make all cores available to our task
#SBATCH --cpus-per-task=1
# Use all partition
#SBATCH --partition=all
# Redirect output and error output
#SBATCH --output=job.out
#SBATCH --error=job.err
source bin/activate
for PLANNER in $(ls benchmark/planner_config)
do
	for SCEN in $(ls benchmark/scenario_config)
	do
		for INDEX in {0..10}
		do
			srun -J "$SCEN@$PLANNER" python -m polygonal_roadmaps -scen $SCEN -planner $PLANNER -index $INDEX 2>&1 &
		done
	done
done
