"""
Jack Farrell, CU Boulder, Dec 2024
Submit 10000 (1000 * 10) SLURM jobs, covering a 100X100 array of parameter space
"""
# imports
import numpy as np
import os
import time


# parameter space
Is = np.linspace(0.000, 2, 100)
n0s = np.linspace(5e15, 1e16, 100)

# other simulation parameters
R = 300
lee = 1e-9
results_dir = f"/scratch/alpine/jafa3629/transport-100x100-8"
stop_wall_time = 2.5

# setup the results directory if it does not already exist
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# parameters for each slurm job
slurm_shebang = "#!/bin/bash\n"
slurm_parameters = f"#SBATCH --account=ucb485_asc1\n#SBATCH --time=02:00:00\n#SBATCH --nodes=1\n#SBATCH --partition=amilan\n#SBATCH --job-name=contriction_transport\n#SBATCH --output={results_dir}/o-%x.%j.out\n"

# submit all jobs
for i in range(len(Is)):
    for j, n0 in enumerate(n0s[::10]):
        
        # each job will run the simulation 10 times
        line1 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j]} --stop-wall-time={stop_wall_time}"
        line2 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+1]} --stop-wall-time={stop_wall_time}"
        line3 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+2]} --stop-wall-time={stop_wall_time}"
        line4 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+3]} --stop-wall-time={stop_wall_time}"
        line5 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+4]} --stop-wall-time={stop_wall_time}"
        line6 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+5]} --stop-wall-time={stop_wall_time}"
        line7 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+6]} --stop-wall-time={stop_wall_time}"
        line8 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+7]} --stop-wall-time={stop_wall_time}"
        line9 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+8]} --stop-wall-time={stop_wall_time}"
        line10 = f"python -um nozzle_1d_finite_volume --results-dir={results_dir} --R={R} --I={Is[i]} --lee={lee} --n0={n0s[10*j+9]} --stop-wall-time={stop_wall_time}"
        
        # build a jobscript with slurm_parameters, and the lines to run
        with open("jobscript.sh", "w") as f:
            f.write(slurm_shebang)
            f.write(slurm_parameters)
            f.write("module load anaconda\n")
            f.write("source /home/jafa3629/.bashrc\n")
            f.write(f"{line1} ; {line2} ; {line3} ; {line4} ; {line5} ; {line6} ; {line7} ; {line8} ; {line9} ; {line10}")

        time.sleep(0.1)
        os.system(f"sbatch jobscript.sh")