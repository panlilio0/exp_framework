"""
Plot fitness over generations given output.csv.
Takes command line arg of csv filename.

Ex: `python3 plot_fitness_over_gens.py --filename output.csv`

Author: Thomas Breimer
February 10th, 2025
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='RL')

parser.add_argument(
    '--filename',  #headless, screen, video, both h, s, v, b
    help='name of csv file',
    default="output.csv")

args = parser.parse_args()

filename = args.filename

df = pd.read_csv(os.path.join("data", filename))

plt.figure(figsize=(8, 5))
plt.plot(np.array(df["generation"]),
         np.array(df["best_so_far"]),
         marker='o',
         linestyle='-',
         color='b',
         label="Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness Evolution Over Generations")
plt.legend()
plt.grid()
plt.show()
