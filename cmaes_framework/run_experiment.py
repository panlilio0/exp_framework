"""
Runs the latest experiment. Also launches best_individual.py to show the best individual so far
and plots fitness over generations after the run is over.

Whether to show the simulation or save as video, number of generations, sigma can be passed as
command line arguments. Example: `python3 run_experiment.py --gens 50 --sigma 2 --mode h` 
runs cma-es for 50 generations in headless mode with a sigma of 2. Replacing "--mode h" with 
"--mode s" makes the simulation output to the screen, and replacing it with "--mode v" saves 
each simulation as a video in `./videos`.  "--mode b" shows on screen and saves a video.

By Thomas Breimer
March 6th, 2025
"""

import argparse
import multiprocessing
import time

import run_cmaes
import best_individual_latest
import plot_fitness_over_gens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--mode', choices=("h","s","v","b"), default="s",
                        help='h=headless, s=screen, v=video, b=both')
    parser.add_argument('--gens', type=int, default=100,
                        help='number of generations to run')
    parser.add_argument('--sigma', type=float, default=1,
                        help='sigma value for CMA-ES')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[2], 
                        help='list of hidden layer sizes')
    args = parser.parse_args()

    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.apply_async(run_cmaes.run(args.mode, args.gens, args.sigma))
        time.sleep(2)
        pool.apply_async(best_individual_latest.visualize_best("t1", args.mode, "false"))
        pool.close()
        pool.join()

    # Plot fitness
    # plot_fitness_over_gens.plot()
