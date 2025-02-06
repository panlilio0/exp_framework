"""
Runs cma-es on `run_simulation.py` as a fitness function.
Continually updates output.csv with best individual from each generation.

Author: Thomas Breimer
February 4th, 2025
"""

import sys
import csv
import run_simulation as sim
from cmaes import CMA
import numpy as np

def run_cma_es(gens, sigma_val):
    """
    Runs the cma_es algorithm on the robot locomotion problem,
    with sin-like robot actuators. Saves a csv file to ./output
    with each robot's genome & fitness for every generation.

    Parameters:
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

    optimizer = CMA(mean=np.array([sim.AVG_FREQ, sim.AVG_AMP, sim.AVG_PHASE_OFFSET] * sim.NUM_ACTUATORS),
                    sigma=sigma_val)

    for generation in range(gens):
        solutions = []

        for indv_num in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = sim.run(sim.NUM_ITERS, x, False)
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
    args = sys.argv

    if len(args) > 1:
        sim.NUM_GENS = int(args[1])

    if len(args) > 2:
        sim.SIGMA = float(args[2])

    run_cma_es(sim.NUM_GENS, sim.SIGMA)




