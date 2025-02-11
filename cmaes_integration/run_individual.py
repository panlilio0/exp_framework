"""
Run a single individual from its genome in an output csv file.
Takes one command line arg corresponding to generation number.
Second command line argument tells whether to show simulation, save it to
video, or both. "screen" renders the video to the screen. "video" saves a
video to the "./videos" folder. "both" does both of these things.

Example: `python3 run_individual.py 10 screen`

Author: Thomas Breimer
January 29th, 2025
"""

import os
import sys
import pathlib
import time
import pandas as pd
import run_simulation

ITERS = 200
FILENAME = "output.csv"

def run_indvididual(generation, mode):
    """
    Run an individual from a csv file.
    
    Parameters:
        generation (int): Generation number of individual.
        mode (string): Tells whether to show simulation, save it to
                       video, or both. "screen" renders the video to the screen. "video" saves a
                       video to the "./videos" folder. "both" does both of these things.
    """

    if mode == "video" or mode == "both":
        os.makedirs("videos", exist_ok=True)

    this_dir = pathlib.Path(__file__).parent.resolve()
    df = pd.read_csv(os.path.join(this_dir, FILENAME))
    row = df.loc[(df['generation']==generation)]
    genome = row.values.tolist()[0][2:]

    run_simulation.run(ITERS, genome, mode, str(int(time.time())))

if __name__ == "__main__":
    args = sys.argv

    if len(args) < 3:
        print("Too few arguments!")
    else:
        run_indvididual(int(args[1]), args[2])
