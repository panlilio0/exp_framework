"""
Plot fitness over generations given output.csv.

Author: Thomas Breimer
February 10th, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "output.csv"

df = pd.read_csv("output.csv")

plt.figure(figsize=(8, 5))
plt.plot(np.array(df["generation"]), np.array(df["best_fitness"]), marker='o', linestyle='-', color='b', label="Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness Evolution Over Generations")
plt.legend()
plt.grid()
plt.show()