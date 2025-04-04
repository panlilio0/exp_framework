"""
Plot genome values on the x and y axis and color based on fitness.
Takes three command line arguments: csv filename, x-axis name, y-axis name.
Example: `python3 plot.py run_1738640858.csv frequency0 amplitude0`

Author: Thomas Breimer
January 29th, 2025
"""

import os
import sys
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

args = sys.argv

if len(args) < 4:
    print("Too few arguments!")
    sys.exit()

filename = args[1]
x_axis_name = args[2]
y_axis_name = args[3]

this_dir = pathlib.Path(__file__).parent.resolve()
path = os.path.join(this_dir, "out", filename)
print(path)
df = pd.read_csv(path)

# Extract relevant columns
x_values = df[x_axis_name]
y_values = df[y_axis_name]
fitness = df["Fitness"]

# Create scatter plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(x_values, y_values, c=fitness, cmap="viridis", edgecolors="k", alpha=0.75)

# Add color bar
cbar = plt.colorbar(sc)
cbar.set_label("Fitness Score")

plt.xlabel("Genome Value " + x_axis_name)
plt.ylabel("Genome Value " + y_axis_name)
plt.title("Genetic Algorithm Population")

# Show plot
plt.show()
