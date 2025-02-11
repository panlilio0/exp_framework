"""
Visualize the best individual so far, pulling from output.csv.

Author: James Gaskell
February 6th, 2025
"""

import pandas
import run_simulation

ITERS = 200
FILENAME = "output.csv"

def visualize_best():

    """
    Look at output.csv and continuously run best individual.
    Assumes csv names are their best achieved fitnesses
    Continually searches for the lowest best fitness, plays the visualization and repeats
    """

    while True:
        df = pandas.read_csv(FILENAME)
        best_fitness = min(df["best_fitness"])
        row = df.loc[df['best_fitness']==best_fitness]
        genome = row.values.tolist()[0][2:]
        print(len(genome))
        run_simulation.run(ITERS, genome, "screen")

if __name__ == "__main__":
    visualize_best()
