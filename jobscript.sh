#!/bin/bash
#SBATCH --account=ucb485_asc1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=amilan
#SBATCH --job-name=contriction_transport
#SBATCH --output=/scratch/alpine/jafa3629/transport-100x100-cartoon-3/5.00000000e+15/o-%x.%j.out
module load anaconda
source /home/jafa3629/.bashrc
python -um nozzle_1d_finite_volume --results-dir=/scratch/alpine/jafa3629/transport-100x100-cartoon-3/5.00000000e+15 --R=3000 --I=2.0 --lee=1e-09 --n0=5000000000000000.0 --stop-wall-time=20