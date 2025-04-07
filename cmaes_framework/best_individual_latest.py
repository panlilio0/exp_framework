"""
Visualize the best individual so far during a run, pulling from latest.csv.

Author: James Gaskell, Thomas Breimer
April 3rd, 2025
"""

import os
from pathlib import Path
import pandas
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
from snn_sim.run_simulation import run

ITERS = 1000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PARENTDIR = Path(__file__).parent.parent.resolve()
GENOME_START_INDEX = 3
FILEFOLDER = "data"

def visualize_best(graphs, filename="latest.csv"):
    """
    Look at a csv and continuously run best individual.
    
    Parameters:
        filename (str): Filename of csv to look at. Defaults to latest.csv.
    """

    #time.sleep(10)


    print(PARENTDIR)
    path = os.path.join(PARENTDIR, filename)

    while True:
        if os.path.exists(path):
            df = pandas.read_csv(path)

            best_fitness = min(df["best_fitness"])
            row = df.loc[df['best_fitness'] == best_fitness]
            genome = row.values.tolist()[0][GENOME_START_INDEX:]

            _, spikes, levels = run(ITERS, genome, "s")

            spikes = np.array(spikes)
            levels = np.array([[x[0] for x in row] for row in levels])

        if graphs == "s":

            fig, ax = plt.subplots(figsize=(12, 5))

            for neuron_idx in range(spikes.shape[1]):
                spike_times = np.where(spikes[:, neuron_idx])[0]
                ax.vlines(spike_times, neuron_idx - 0.4, neuron_idx + 0.4)

            ax.set_yticks(np.arange(spikes.shape[1]))
            ax.set_yticklabels([f'Neuron {i}' for i in range(spikes.shape[1])])
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Neuron Index")
            ax.set_title("Spike Train of Neurons")
            plt.tight_layout()
            plt.xlim(0, 100)
            plt.show(block=True)
        
        elif graphs == "l":

            fig, ax = plt.subplots(figsize=(12, 6))

            for i in range(levels.shape[1]):
                ax.plot(levels[:, i], label=f'Neuron {i}')

            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Level')
            ax.set_title('Neuron Levels Over Time')
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.show(block=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--graphs',
        help='graph outputs and levels? s - spike trains, l - levels, y - both',
        default="l")
    
    args = parser.parse_args()
    
    visualize_best(args.graphs)

        
    