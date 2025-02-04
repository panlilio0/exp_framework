"""
Plot two genome values with color-based fitness & generation slider.
Takes three command line arguments: csv filename, x-axis name, y-axis name.
Example: `python3 plot_gens.py run_1738640858.csv frequency0 amplitude0`

Author: Thomas Breimer
January 29th, 2025
"""

import os
import sys
import pathlib
import plotly.express as px
import pandas as pd

args = sys.argv

if len(args) < 4:
    print("Too few arguments!")
    sys.exit()

filename = args[1]
x_axis_name = args[2]
y_axis_name = args[3]

this_dir = pathlib.Path(__file__).parent.resolve()
df = pd.read_csv(os.path.join(this_dir, "out", filename))

# Create scatter plot
fig = px.scatter(
    df,
    x=df[x_axis_name],
    y=df[y_axis_name],
    color="Fitness",
    color_continuous_scale="viridis",
    animation_frame="Generation",
    animation_group="Individual",
    title="Evolution of CMA-ES Over Generations",
    hover_data=df.columns
)

fig.show()
