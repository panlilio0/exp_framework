"""
Visualize the best individual so far, pulling from output.csv.

Author: James Gaskell
February 6th, 2025
"""

import os
import argparse
import pandas
from snn_sim.run_simulation import run

ITERS = 1000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def visualize_best(filename):
    """
    Look at output.csv and continuously run best individual.
    Assumes csv names are their best achieved fitnesses
    Continually searches for the lowest best fitness, plays the visualization and repeats
    """

    while True:
        path = os.path.join(ROOT_DIR, "data", filename)
        df = pandas.read_csv(path)
        best_fitness = min(df["best_fitness"])
        row = df.loc[df['best_fitness'] == best_fitness]
        genome = row.values.tolist()[0][2:]
        run(ITERS, genome, "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--file',
                        type=str,
                        default=None,
                        help='csv file to run')
    
    args = parser.parse_args()

    if args.file == None:
        raise Exception('No csv file specified!')

    visualize_best(args.file)
