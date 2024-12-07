from .solver import *
from .config import *
import argparse

######### command line interface
def parse_args():
    parser = argparse.ArgumentParser(description="2D channel flow simulation")
    parser.add_argument("--n1", type=float, default=n1, help="Fixed density at the left boundary")
    parser.add_argument("--n2", type=float, default=n2, help="Fixed density at the right boundary")
    parser.add_argument("--results-dir", default=results_dir, help="Directory to save the results")
    parser.add_argument("--eta", type=float, default=eta, help="Viscosity")
    parser.add_argument("--gamma", type=float, default=gamma, help="Momentum relaxation rate")
    parser.add_argument("--draw_plot", type=bool, default=draw_plot, help="Whether to draw plots")
    parser.add_argument("--stop_wall_time", type=float, default=stop_wall_time, help="CPU hours to run for")
    parser.add_argument("--save_after", type=float, default=save_after, help="CPU hours to wait before starting to save data")
    parser.add_argument("--save_increment", type=float, default=save_increment, help="Simulation time between snapshots to save")
    parser.add_argument("--h", type=float, default=h, help="Grid spacing in x direction")
    parser.add_argument("--hy", type=float, default=hy, help="Grid spacing in y direction")
    parser.add_argument("--k", type=float, default=k, help="Time step")
    return parser.parse_args()
args = parse_args()

########## run simulation
solve(
    n1=args.n1,
    n2=args.n2,
    results_dir=args.results_dir,
    eta=args.eta,
    gamma=args.gamma,
    draw_plot=args.draw_plot,
    stop_wall_time=args.stop_wall_time,
    save_after=args.save_after,
    save_increment=args.save_increment,
    h=args.h,
    hy=args.hy,
    k=args.k
)
