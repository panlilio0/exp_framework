'''
plotting csv files for james.
'''

import os
import argparse
import glob
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PARENTDIR = Path(__file__).parent.resolve()
FILEFOLDER = "distance_outputs"

def plot_all(csv_filename):
    '''
    Given a csv with distances, plots the distances.
    '''
    #read
    folder_path = os.path.join(PARENTDIR, FILEFOLDER)

    df = pd.read_csv(folder_path + "/" + csv_filename)

    #start plot
    plottitle = csv_filename[:-4] + " actions"
    plt.title(plottitle)
    plt.ylabel("distances")
    plt.xlabel("steps")

    # get columns
    cols = df.columns.tolist()
    for col in cols:
        to_plot = list(df[col])
        plt.plot(range(len(to_plot)), to_plot, label=col)

    plt.legend(loc='upper center')
    plt.savefig(os.path.join(folder_path, plottitle + '.png'))
    plt.close()

def get_latest_file():
    """
    Finds the most recently created file in a folder matching a given pattern.
    
    Returns:
        str: The path to the most recently created file, or None if no files match the pattern.
    """
    this_dir = Path(__file__).parent.resolve()
    files = glob.glob(os.path.join(this_dir, FILEFOLDER) + "/*.csv")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime).split("/")[-1]
    return latest_file


if __name__ == "__main__":

    #args
    PARSER = argparse.ArgumentParser(description="Arguments for simple experiment.")
    PARSER.add_argument("--csv_filename", type=str, default=get_latest_file(),
                        help="The csv file of the robot distances you want to plot.")

    ARGS = PARSER.parse_args()

    plot_all(ARGS.csv_filename)
