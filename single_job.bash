#!/bin/bash
#SBATCH --time=10:00
# Make all cores available to our task
#SBATCH --cpus-per-task=1
# Use all partition
#SBATCH --partition=ci
# Redirect output and error output
#SBATCH --output="logs/slurm-%A-%a.out"
#SBATCH --error="logs/slurm-%A-%a.err"
MAP=$2
PLANNER=$1
source bin/activate
srun python -m polygonal_roadmaps -maps $MAP -planner $PLANNER -index $SLURM_ARRAY_TASK_ID -logfile "logs/$PLANNER$MAP/$SLURM_ARRAY_TASK_ID.log" -loglevel warning
