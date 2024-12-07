#!/bin/bash
#SBATCH --account=ucb485_asc1
#SBATCH --partition=amilan
#SBATCH --job-name=constriction
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=14:20:00
module purge
module load python
module load anaconda

python -u solver.py
