"""
Run a single individual from its genome in an output csv file.
Takes three command line args: filename, generation num, individual num.
Example: `python3 run_2378234.csv 4 13`

Author: Thomas Breimer
January 29th, 2025
"""

import os
import sys
import pathlib
import pandas as pd
import run_cma_es

ITERS = 200

def run_indvididual(filename, generation, individual):
    """
    Run an individual from a csv file.
    
    Parameters:
        filename (str): Name of csv file.
        generation (int): Generation number of individual.
        individual (int): Number of individual in generation.
    """

    this_dir = pathlib.Path(__file__).parent.resolve()
    df = pd.read_csv(os.path.join(this_dir, "out", filename))
    row = df.loc[(df['Generation']==generation) & (df['Individual']==individual)]
    genome = row.values.tolist()[0][3:]

    run_cma_es.run_simulation(ITERS, genome)

if __name__ == "__main__":
    args = sys.argv

    if len(args) < 4:
        print("Too few arguments!")
    else:
        run_indvididual(str(args[1]), int(args[2]), int(args[3]))
