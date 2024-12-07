import os
import numpy as np
import time

# voltage values
V = np.arange(2,12,0.01)
N = V.shape[0]


# convert V to a certain value of n2, assuming n2 - n1 = C*V/e...
# NOTE given recent conversation with Johannes, is relationship between n and V more complicated?
e = 1.602e-19
hbar = 1.0456e-34
m = 9.109e-31 * 0.06 # effective mass
L = 40e-6 # 40 micronsls
W = 0.342 * L

# gate parameters
d = 307e-9 # meters
e0 = 8.854e-12
er = 3.9
C = e0*er/d

VG = -7 # volts
CNP = -10 # volts

# density of carriers at drain
n0 = np.abs(C*(VG - CNP)/(-e))

dn = (C*V/e) / n0

# holes need to move right, toward drain
n1s = 1+dn # on left end
n2 = 1

# calculate v_s evaluated at at the fixed boundary, with capacitive effects
vs = np.sqrt((hbar/m*np.sqrt(np.pi*n0/2))**2 + (n0*e**2/m/C))

# resistance of the channe
R = 3000 # Ohms
rho = R * W / L * 0.29 # resistivity, with appropriate geometric prefactor since sample is not rectangle

# estimate tau_mr, gamma_mr in simulation units with drude formula
tau = m / (n0*e**2) * vs / L / rho
gamma = 1/tau
#print(gamma)

# configure and run the simulation
# using a dictionary to pass parameters to solver, so we don't have to write solve(k=k, h=h, hy=hy, n2=n2, n1=n1, gamma=gamma, eta=eta, results_dir=results_dir, save_after=save_after, save_increment=save_increment, stop_wall_time=stop_wall_time)
# configure

# do some stuff
if __name__ == "__main__":

    # create runtables
    with open("runtable.dat", "w") as f:
        for n1 in n1s:
            f.write(f"python -um nozzle_sim --n1 {n1} --gamma {gamma} --results-dir /scratch/alpine/jafa3629/currentsweep4\n")




    # construct and build a job script to run the array job
    with open("current.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --account=ucb485_asc1\n")
        f.write("#SBATCH --partition=amilan\n")
        f.write("#SBATCH --job-name=constriction\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --mem=2G\n")
        f.write("#SBATCH --time=05:00:00\n")
        f.write(f"#SBATCH --array=1-{N}\n")
        f.write("module load anaconda\n")
        f.write("source /home/jafa3629/.bashrc\n")
        f.write("echo $(cat runtable.dat | head -${SLURM_ARRAY_TASK_ID} | tail -1)\n")
        f.write('eval $(cat runtable.dat | head -${SLURM_ARRAY_TASK_ID} | tail -1)')
        
    time.sleep(1)

    # run the job script
    os.system("sbatch current.sh")
