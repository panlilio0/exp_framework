"""
Plot genome values on the x and y axis and color based on fitness.
Takes three command line arguments: --filename, --x-axis name, --y-axis.

Author: Thomas Breimer
January 29th, 2025
"""

import os
import argparse
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='RL')

parser.add_argument(
    '--filename',  #headless, screen, video, both h, s, v, b
    help='name of csv file',
    default="output.csv")
parser.add_argument('--xaxis',
                    type=str,
                    help='what genome element to go on the x-axis',
                    default="weight0")
parser.add_argument('--yaxis',
                    type=str,
                    help='what genome element to go on the y-axis',
                    default="weight1")

args = parser.parse_args()

filename = args.filename
x_axis_name = args.xaxis
y_axis_name = args.yaxis

this_dir = pathlib.Path(__file__).parent.resolve()
path = os.path.join(this_dir, "data", filename)
df = pd.read_csv(path)

# Extract relevant columns
x_values = df[x_axis_name]
y_values = df[y_axis_name]
fitness = df["best_fitness"]

# Create scatter plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(x_values,
                 y_values,
                 c=fitness,
                 cmap="viridis",
                 edgecolors="k",
                 alpha=0.75)

# Add color bar
cbar = plt.colorbar(sc)
cbar.set_label("Fitness Score")

plt.xlabel("Genome Value " + x_axis_name)
plt.ylabel("Genome Value " + y_axis_name)
plt.title("Genetic Algorithm Population")

# Show plot
plt.show()
