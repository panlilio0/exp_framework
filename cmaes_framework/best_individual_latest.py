"""
Visualize the best individual so far during a run, pulling from latest.csv.

Author: James Gaskell, Thomas Breimer
April 3rd, 2025
"""

import os
from pathlib import Path
import pandas
import time
from snn_sim.run_simulation import run

ITERS = 1000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PARENTDIR = Path(__file__).parent.parent.resolve()
GENOME_START_INDEX = 3
FILEFOLDER = "data"

def visualize_best(filename="latest.csv"):
    """
    Look at a csv and continuously run best individual.
    
    Parameters:
        filename (str): Filename of csv to look at. Defaults to latest.csv.
    """

    #time.sleep(10)


    print(PARENTDIR)
    path = os.path.join(PARENTDIR, filename)

    while True:
        if os.path.exists(path):
            df = pandas.read_csv(path)

            best_fitness = min(df["best_fitness"])
            row = df.loc[df['best_fitness'] == best_fitness]
            genome = row.values.tolist()[0][GENOME_START_INDEX:]

            run(ITERS, genome, "s")

if __name__ == "__main__":
    visualize_best()

        
    