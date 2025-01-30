"""
Run a single individual from its genome in an output csv file.

Author: Thomas Breimer
January 29th, 2025
"""

import os
import run_cma_es
import pathlib
import pandas as pd

individual = 2
generation = 10
iters = 200
filename = "2025-01-29 17:03:02.626331_run.csv"

this_dir = pathlib.Path(__file__).parent.resolve()    
df = pd.read_csv(os.path.join(this_dir, "out", filename))
row = df.loc[(df['Generation']==generation) & (df['Individual']==individual)]
genome = row.values.tolist()[0][3:]

fitness = run_cma_es.run_simulation(iters, genome)

