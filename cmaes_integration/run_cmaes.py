"""
Runs cma-es on `run_simulation.py` as a fitness function.
Creates output.csv and updates it continuously with the best individual from each generation.
Whether to show the simulation or save as video, number of generations, sigma can be passed as
command line arguments. Example: `python3 run_cmaes.py headless 50 2` runs cma-es for 50 generations
in headless mode with a sigma of 2. Replacing "headless" with "screen" makes the simulation 
output to the screen, and replacing it with "video" saves each simulation as a video in `./videos`. 
"both" shows on screen and saves a video.

Author: Thomas Breimer
February 4th, 2025
"""

import sys, csv, argparse
from cmaes import CMA
import numpy as np
import run_simulation as sim

def run_cma_es(mode, gens, sigma_val):
    """
    Runs the cma_es algorithm on the robot locomotion problem,
    with sin-like robot actuators. Saves a csv file to ./output
    with each robot's genome & fitness for every generation.

    Parameters:
        mode (string): How to run the simulation. 
                       "headless" runs without any video or visual output.
                       "video" outputs the simulation as a video in the "./videos folder.
                       "screen" shows the simulation on screen as a window.
                       "both: shows the simulation on a window and saves a video.
        gens (int): How many generations to run.
        sigma_val (float): The standard deviation of the normal distribution
        used to generate new candidate solutions
    """

    # Generate output.csv file

    csv_header = ['generation', 'best_fitness']

    for i in range(sim.NUM_ACTUATORS):
        csv_header = csv_header + ['frequency' + str(i), 'amplitude' + str(i),
                             'phase_offset' + str(i)]

    with open("output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

    # Init CMA

    optimizer = CMA(mean=np.array([sim.AVG_FREQ, sim.AVG_AMP, sim.AVG_PHASE_OFFSET]
                                   * sim.NUM_ACTUATORS), sigma=sigma_val)

    for generation in range(gens):
        solutions = []

        for indv_num in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = sim.run(sim.NUM_ITERS, x, mode, str(generation) + "_" + str(indv_num))
            solutions.append((x, fitness))

        optimizer.tell(solutions)
        print([i[1] for i in solutions])
        print("Generation", generation, "Best Fitness:", solutions[0][1])

        # Add a new row to output.csv file with cols: generation#, fitness, and genome
        new_row = [generation, solutions[0][1]] + solutions[0][0].tolist()

        with open("output.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(new_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode', #headless, screen, video, both h, s, v, b
        help='number of generations to run',
        default="h")
    parser.add_argument(
        '--gens',
        help='number of generations to run',
        default=100)
    parser.add_argument(
        '--sigma',
        default=2,
        help='sigma value for cma-es')
    args = parser.parse_args()

    #sim.MODE = args.mode
    #sim.NUM_GENS = args.gens
    #sim.SIGMA = args.sigma

    run_cma_es(args.mode, int(args.gens), int(args.sigma))