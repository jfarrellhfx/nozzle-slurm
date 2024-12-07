import numpy as np

########## configuration
# setup, unlikely to change
LX, LY = 1.0, 0.342 # aspect ratio based on device geometry

# discretizations
h = LX/200
hy = LY/200
k = 0.0001

# how long to simulate
stop_wall_time = 1 # cpu-hours to run for

# saving data
results_dir = "good-copies"
save_increment = 0.01 # simtime between snapshots
save_after = 0 # how long to wait before starting to save data

# physical parameters
eta = 0.008 # viscosity
gamma = 1.00 # momentum relaxation rate

# whether or not to draw the plot
draw_plot = False



# boundary conditions
n1 = 3 # left
n2 = 1 # right
