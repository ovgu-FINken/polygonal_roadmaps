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
SCENARIO=$1
N_AGENTS=$2
PLANNER=$3

source /opt/spack/main/env.sh
module load python
source bin/activate
srun python -m polygonal_roadmaps -n_agents $N_AGENTS -index $SLURM_ARRAY_TASK_ID \
    -logfile "logs/$PLANNER$SCENARIO/$SLURM_ARRAY_TASK_ID.log" -loglevel warning -memlimit 5 -timelimit 30 \
    -planner $PLANNER -scenario $SCENARIO
