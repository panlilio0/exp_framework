'''
plotting csv files for james.
'''

import os
import argparse
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILEFOLDER = "distance_outputs"

def plot_all(csv_filename):
    '''
    Given a csv with distances, plots the distances.
    '''
    #read
    df = pd.read_csv(csv_filename)

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
    plt.savefig(os.path.join(FILEFOLDER, plottitle + '.png'))
    plt.close()



if __name__ == "__main__":

    #args
    PARSER = argparse.ArgumentParser(description="Arguments for simple experiment.")
    PARSER.add_argument("csv_filename", type=str,
                        help="The csv file of the robot distances you want to plot.")

    ARGS = PARSER.parse_args()

    plot_all(ARGS.csv_filename)
