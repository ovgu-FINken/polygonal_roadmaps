#!/bin/bash
#SBATCH --time=1:00:00
# Make all cores available to our task
#SBATCH --cpus-per-task=1
# Use all partition
#SBATCH --partition=ci
#SBATCH --mem-per-cpu=4Gb
# Redirect output and error output
#SBATCH --output="logs/slurm.out"
#SBATCH --error="logs/slurm.err"
MAP=$2
PLANNER=$1
source bin/activate
srun python -m polygonal_roadmaps -maps $MAP -planner $PLANNER -index $SLURM_ARRAY_TASK_ID -logfile "logs/$PLANNER$MAP/$SLURM_ARRAY_TASK_ID.log" -loglevel warning -memlimit 2 -timelimit 5
