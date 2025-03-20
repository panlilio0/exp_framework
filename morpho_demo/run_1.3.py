"""
Simple RMHC of walking robot from scratch in Evogym with internal time scale tracking.

Author: Thomas Breimer, Matthew Meek
Modified by: Darren
Date: February 2025
"""

import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject

# Simulation parameters
ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 10
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
FRAME_CYCLE_LEN = 10
NUM_ACTUATORS = 5
NUM_ITERS = 100
MUTATE_RATE = 0.2
GENS = 1500
STOPPING_THRESHOLD = 0.0005
NO_IMPROVEMENT_LIMIT = 50

ENV_FILE_NAME = "simple_environment_long.json"
ROBOT_FILE_NAME = "walkbot4billion_reduced.json"
EXPER_DIR = 'score_plots/' + ROBOT_FILE_NAME[:-5] + " " + time.asctime()


def run_rmhc(gens, show=True):
    """Run RMHC in Evogym and track simulation time scales."""
    os.mkdir(EXPER_DIR)

    iters = NUM_ITERS
    genome = np.random.rand(NUM_ACTUATORS * 2)

    fitness_progress = []
    generation_times = []

    best_fitness, _ = run_simulation(iters, genome, show)
    print(f"Starting fitness: {best_fitness:.6f}")

    no_improvement_count = 0
    for i in range(gens):
        mutated_genome = genome.copy()
        mutated_genome = np.array([
            random.random() if random.random() < MUTATE_RATE else x
            for x in mutated_genome
        ])

        new_fitness, avg_velocity = run_simulation(iters, mutated_genome, show)

        fitness_progress.append(new_fitness)
        generation_times.append(i)

        if new_fitness > best_fitness:
            print(
                f"🟢 Found better fitness after {i} generations: {new_fitness:.6f}"
            )
            best_fitness = new_fitness
            genome = mutated_genome
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Stop if fitness has not improved for NO_IMPROVEMENT_LIMIT generations
        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            print(
                f"🛑 Stopping early: No fitness improvement for {NO_IMPROVEMENT_LIMIT} generations"
            )
            break

        # Stop if velocity is too low
        if avg_velocity < STOPPING_THRESHOLD:
            print(
                f"🛑 Stopping early: Robot velocity is too low ({avg_velocity:.6f})"
            )
            break

    print(f"\nFinal fitness after {i+1} generations: {best_fitness:.6f}")

    log_fitness_progress(fitness_progress, generation_times)
    run_simulation(NUM_ITERS * 5, genome, fittest=True)

    return genome, best_fitness


def run_simulation(iters, genome, show=True, fittest=False):
    """Runs a simulation and tracks internal time scales."""
    world = EvoWorld.from_json(os.path.join('world_data', ENV_FILE_NAME))
    robot = WorldObject.from_json(os.path.join('world_data', ROBOT_FILE_NAME))

    world.add_from_array(name='robot',
                         structure=robot.get_structure(),
                         x=ROBOT_SPAWN_X,
                         y=ROBOT_SPAWN_Y,
                         connections=robot.get_connections())

    sim = EvoSim(world)
    sim.reset()
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    fitness = 0
    prev_com = None
    total_velocity = 0
    velocity_over_time = []

    for step in range(iters):
        pos_1 = sim.object_pos_at_time(sim.get_time(), "robot")
        com_1 = np.mean(pos_1, 1)

        action = genome * ((com_1[0] + com_1[1]) / 2)
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        if len(action) > NUM_ACTUATORS:
            action = action[:NUM_ACTUATORS]
        elif len(action) < NUM_ACTUATORS:
            action = np.concatenate(
                [action,
                 np.zeros(NUM_ACTUATORS - len(action))])  # Pad with zeros

        sim.set_action('robot', action)
        sim.step()

        pos_2 = sim.object_pos_at_time(sim.get_time(), "robot")
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
        fitness += reward

        if prev_com is not None:
            velocity = abs(com_2[0] - prev_com[0])
            total_velocity += velocity
            velocity_over_time.append(velocity)

        prev_com = com_2

        if show:
            viewer.render('screen', verbose=True)

    viewer.close()
    avg_velocity = total_velocity / iters

    log_velocity_time(velocity_over_time)
    return fitness, avg_velocity


def log_fitness_progress(fitness_progress, generation_times):
    """Save fitness progress over generations."""
    df = pd.DataFrame({
        'Generation': generation_times,
        'Fitness': fitness_progress
    })
    df.to_csv(os.path.join(EXPER_DIR, "fitness_progress_log.csv"), index=False)
    plt.plot(generation_times,
             fitness_progress,
             label="Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig(os.path.join(EXPER_DIR, "fitness_progress_plot.png"))
    plt.close()


def log_velocity_time(velocity_over_time):
    """Save velocity progression over simulation steps."""
    df = pd.DataFrame({
        'Step': range(len(velocity_over_time)),
        'Velocity': velocity_over_time
    })
    df.to_csv(os.path.join(EXPER_DIR, "velocity_time_log.csv"), index=False)
    plt.plot(range(len(velocity_over_time)),
             velocity_over_time,
             label="Velocity Over Steps")
    plt.xlabel("Simulation Step")
    plt.ylabel("Velocity")
    plt.legend()
    plt.savefig(os.path.join(EXPER_DIR, "velocity_time_plot.png"))
    plt.close()


if __name__ == "__main__":
    run_rmhc(GENS, False)
