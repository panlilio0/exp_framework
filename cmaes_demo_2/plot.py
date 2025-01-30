"""
Plot genome values on the x and y axis and color based on fitness.

Author: Thomas Breimer
January 29th, 2025
"""

import os
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

filename = "2025-01-29 17:03:02.626331_run.csv"

this_dir = pathlib.Path(__file__).parent.resolve()    
df = pd.read_csv(os.path.join(this_dir, "out", filename))

# Choose genome indices for x and y axes
x_index = 4
y_index = 5 

# Extract relevant columns
x_values = df.iloc[:, x_index]
y_values = df.iloc[:, y_index]
fitness = df["Fitness"]

# Create scatter plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(x_values, y_values, c=fitness, cmap="viridis", edgecolors="k", alpha=0.75)

# Add color bar
cbar = plt.colorbar(sc)
cbar.set_label("Fitness Score")

plt.xlabel(f"Genome Value {x_index-3}")
plt.ylabel(f"Genome Value {y_index-3}")
plt.title("Genetic Algorithm Population")

# Show plot
plt.show()