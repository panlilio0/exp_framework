"""
Code for making plots based on SNN logs.

Author: Thomas Breimer
Modified by: James Gaskell

Last modified:
April 16th, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import *
from pathlib import Path

def load_logs(file_path):
    """Load CSV log data."""
    return pd.read_csv(file_path)


def plot_neuron_logs(df, xlim, snn_id, layer, neuron_id):
    """Plot logs for a specific neuron from dataframe."""
    
    neuron_df = df[(df['SNN'] == snn_id) &
                   (df['layer'] == layer) &
                   (df['neuron'] == neuron_id)]

    if neuron_df.empty:
        print("No data found for this neuron.")
        return

    logs = {}
    for log_type in ['levellog', 'firelog', 'dutycyclelog']:
        row = neuron_df[neuron_df['log'] == log_type]
        if not row.empty:
            logs[log_type] = row.iloc[0, 4:].astype(float).values
        else:
            logs[log_type] = None

    steps = range(len(logs['levellog']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    if logs['levellog'] is not None:
        ax1.plot(steps, logs['levellog'], label='Activation Level (levellog)', alpha=0.7)

    if logs['firelog'] is not None:
        spike_steps = [i for i, v in enumerate(logs['firelog']) if v > 0]
        ymin, ymax = ax1.get_ylim()
        spike_ymin = ymin + 0.05 * (ymax - ymin)
        spike_ymax = spike_ymin + 0.1 * (ymax - ymin)
        for spike in spike_steps:
            ax1.vlines(x=spike, ymin=spike_ymin, ymax=spike_ymax, color='red', linewidth=1.5)

    ax1.set_ylabel('Activation Level')
    ax1.set_title(f'SNN {snn_id} | Layer: {layer} | Neuron: {neuron_id}')
    ax1.legend()
    ax1.grid(True)

    if logs['dutycyclelog'] is not None:
        ax2.plot(steps, logs['dutycyclelog'], label='Duty Cycle (dutycyclelog)', color='purple', alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Duty Cycle')
    ax2.set_xlabel('Timestep')
    ax2.legend()
    ax2.grid(True)

    plt.xlim(0, xlim)
    plt.tight_layout()
    plt.show()

def plot_snn_spiketrains(df, xlim, snn_id):
    """Plot spike trains for all neurons in an SNN with colored lines per neuron."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'firelog')]
    snn_df = snn_df.sort_values(by=['layer', 'neuron']).reset_index(drop=True)

    num_neurons = len(snn_df)
    palette = sns.color_palette("husl", num_neurons)

    plt.figure(figsize=(12, num_neurons * 0.5 + 2))

    yticks = []
    yticklabels = []

    for idx, (_, row) in enumerate(snn_df.iterrows()):
        spikes = row.iloc[4:].astype(float).values
        spike_steps = [i for i, v in enumerate(spikes) if v > 0]
        plt.vlines(spike_steps, idx, idx + 1, color=palette[idx], linewidth=1.5)

        yticks.append(idx + 0.5)
        yticklabels.append(f"{row['layer']}-{row['neuron']}")

    plt.yticks(yticks, yticklabels)
    plt.xlabel('Timestep')
    plt.ylabel('Neuron')
    plt.title(f'SNN {snn_id} Spike Trains')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()

def plot_snn_activation_levels(df, xlim, snn_id):
    """Plot activation levels for all neurons in an SNN."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'levellog')]

    plt.figure(figsize=(12, 6))

    for _, row in snn_df.iterrows():
        levels = row.iloc[4:].astype(float).values
        label = f"{row['layer']}-{row['neuron']}"
        plt.plot(levels, label=label, alpha=0.7)

    plt.title(f'SNN {snn_id} Activation Levels')
    plt.xlabel('Timestep')
    plt.ylabel('Activation Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()


def plot_snn_dutycycles(df, xlim, snn_id):
    """Plot duty cycles for all neurons in an SNN."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'dutycyclelog')]

    plt.figure(figsize=(12, 6))

    for _, row in snn_df.iterrows():
        duty = row.iloc[4:].astype(float).values
        label = f"{row['layer']}-{row['neuron']}"
        plt.plot(duty, label=label, alpha=0.7)

    plt.title(f'SNN {snn_id} Duty Cycles')
    plt.xlabel('Timestep')
    plt.ylabel('Duty Cycle')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()


def plot_fitness_over_time(experiment_path):
    """
    Plot fitness over time across all runs in the given folder.
    - If multiple runs: plot mean, min, and 25-75 percentile shading
    - If single run: plot only best_so_far line
    """

    exp_path = Path(__file__).parent.parent.resolve() / experiment_path
    print(f"Looking for run_*.csv files in: {exp_path}")

    if not exp_path.exists() or not exp_path.is_dir():
        print(f"Path '{experiment_path}' does not exist or is not a directory.")
        return

    run_files = list(exp_path.glob("run_*.csv"))
    num_files = len(run_files)

    if num_files == 0:
        print(f"No run_*.csv files found in {exp_path}")
        return

    all_data = []

    for file in run_files:
        try:
            df = pd.read_csv(file)
            if 'generation' in df.columns and 'best_so_far' in df.columns:
                df = df[['generation', 'best_so_far']]
                df['run'] = file.name
                all_data.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")

    if not all_data:
        print("No valid data to plot.")
        return

    combined = pd.concat(all_data, ignore_index=True)

    plt.figure(figsize=(12, 6))

    avg_color = '#ff871d'  # Orange for average best fitness
    best_color = '#4fafd9'  # Blue for best fitness per generation
    shade_color = '#ffb266'

    if num_files == 1:
        df = all_data[0]
        plt.plot(df['generation'], df['best_so_far'], label="Best Fitness", linewidth=2, color=best_color)
    else:
        stats = combined.groupby('generation')['best_so_far'].agg([
            'mean', 'min', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
        ]).rename(columns={
            '<lambda_0>': 'q25',
            '<lambda_1>': 'q75'
        }).reset_index()

        plt.plot(stats['generation'], stats['mean'], marker='o', label="Mean Best Fitness", linewidth=2, color=avg_color)
        plt.plot(stats['generation'], stats['min'], marker='o', label="Overall Best Fitness", color=best_color)
        plt.fill_between(stats['generation'], stats['q25'], stats['q75'], alpha=0.2, label='25-75th Percentile', color=shade_color)

    plt.xlabel('Generation')
    plt.ylabel('Best Fitness So Far')
    title = (f'Fitness Over Time Across {num_files} Runs')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def boxplot_last_generation_fitness(experiment_path):
    """
    Create a box plot of best fitness values in the last generation across all runs in the given folder.
    """

    exp_path = Path(__file__).parent.parent.resolve() / experiment_path
    print(f"Looking for run_*.csv files in: {exp_path}")

    if not exp_path.exists() or not exp_path.is_dir():
        print(f"Path '{experiment_path}' does not exist or is not a directory.")
        return

    run_files = list(exp_path.glob("run_*.csv"))
    if not run_files:
        print(f"No run_*.csv files found in {exp_path}")
        return

    last_gen_fitnesses = []

    for file in run_files:
        try:
            df = pd.read_csv(file)
            if 'generation' in df.columns and 'best_so_far' in df.columns:
                last_gen = df['generation'].max()
                last_fitness = df[df['generation'] == last_gen]['best_so_far'].values
                if len(last_fitness) > 0:
                    last_gen_fitnesses.append(last_fitness[0])
        except Exception as e:
            print(f"Could not read {file}: {e}")

    if not last_gen_fitnesses:
        print("No valid data found.")
        return

    plt.figure(figsize=(8, 6))
    plt.boxplot(last_gen_fitnesses, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#ff871d', color='#ff871d'),
                medianprops=dict(color='#4fafd9', linewidth=2))
    
    plt.ylabel('Best Fitness (Last Generation)')
    plt.title(f'Best Fitnesses in Last Generation\n(Across {len(last_gen_fitnesses)} Runs)')
    plt.grid(axis='y')
    plt.xticks([1], ['Last Gen'])

    plt.tight_layout()
    plt.show()

