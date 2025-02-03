"""
Plot two genome values with color-based fitness & generation slider.

Author: Thomas Breimer
January 29th, 2025

"""

import os
import pathlib
import plotly.express as px
import pandas as pd

filename = "2025-01-29 17:03:02.626331_run.csv"

this_dir = pathlib.Path(__file__).parent.resolve()    
df = pd.read_csv(os.path.join(this_dir, "out", filename))

# Choose genome indices for x and y axes
x_index = 4  
y_index = 5

# Create scatter plot
fig = px.scatter(
    df, 
    x=df.columns[x_index], 
    y=df.columns[y_index], 
    color="Fitness", 
    color_continuous_scale="viridis",
    animation_frame="Generation",
    animation_group="Individual",
    title="Evolution of CMA-ES Over Generations",
    labels={
        df.columns[x_index]: f"Genome Value {x_index-3}", 
        df.columns[y_index]: f"Genome Value {y_index-3}",
        "color": "Fitness"
    },
    hover_data=df.columns
)

fig.show()