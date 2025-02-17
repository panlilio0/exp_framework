"""

Runs a random mutation hillclimber on 'run_simplation.py'



Author: James Gaskell
February 13th, 2025
"""

import sys, csv, argparse
import run_sim_rmhc as sim
import numpy as np
from datetime import datetime


def run_rmhc(mode, gens):

    csv_header = ['generation', 'best_fitness']
    
    now = datetime.now()
    time = now.strftime("%Y-%m-%d_%H:%M:%S")

    for i in range(sim.NUM_ACTUATORS):
        csv_header = csv_header + ['voxel_' + str(i) + '_genome_val']
    
    with open("output/RMHC_" + time +".csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

    genome = np.random.rand(10)
    fitness = sim.run(sim.NUM_ITERS, genome, mode)

    for generation in range(gens):

        mutated_genome = genome.copy()

        for j in range(len(mutated_genome)):
            if np.random.random() < 0.1:
                mutated_genome[j] = np.random.random()

        new_fitness = sim.run(sim.NUM_ITERS, mutated_genome, mode)

        if new_fitness < fitness:
            print("Found better after", generation, "generations:", new_fitness)
            fitness = new_fitness
            genome = mutated_genome

            with open("output/RMHC_" + time + ".csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([generation] + [fitness] + list(genome))
    
    print("Final fitness", fitness)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode', #headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="h")
    parser.add_argument(
        '--gens',
        type=int,
        help='number of generations to run',
        default=10000)
    args = parser.parse_args()

    run_rmhc(args.mode, args.gens)