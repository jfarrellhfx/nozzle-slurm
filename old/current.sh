#!/bin/bash
#SBATCH --account=ucb485_asc1
#SBATCH --partition=amilan
#SBATCH --job-name=constriction
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --time=05:00:00
#SBATCH --array=1-1000
module load anaconda
source /home/jafa3629/.bashrc
echo $(cat runtable.dat | head -${SLURM_ARRAY_TASK_ID} | tail -1)
eval $(cat runtable.dat | head -${SLURM_ARRAY_TASK_ID} | tail -1)