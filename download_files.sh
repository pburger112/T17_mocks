#!/bin/bash

# Example of running python script in a batch mode


#SBATCH --account=def-mjhudson
#SBATCH --mail-user=pierre.burger@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --array=0-107

# Run python script
module load python/3.10
source /home/pburger/graham_envs/DSL_env/bin/activate
srun python T17_compute_SMF.py $SLURM_ARRAY_TASK_ID