"""
Soft Robotics Latency Tracking in Evogym

Tracks latency: (1) Robot falls (high velocity), (2) First stop, (3) Actuators restart movement, (4) Robot stops again.
Integrates voxel corner tracking from `run_1_arg_corners.py`.

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
from run_1_arg_corners import find_corners, get_all_distances

# Simulation parameters
ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 10
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
FRAME_CYCLE_LEN = 10
NUM_ACTUATORS = 5  
NUM_ITERS = 5000  
MUTATE_RATE = 0.1  
GENS = 1500
STOPPING_THRESHOLD = 0.0005  
NO_IMPROVEMENT_LIMIT = 500 

ENV_FILE_NAME = "simple_environment_long.json"
ROBOT_FILE_NAME = "walkbot4billion_reduced.json"
EXPER_DIR = 'score_plots/' + ROBOT_FILE_NAME[:-5] + " " + time.asctime()

# HARDCODED_ACTIONS = np.array([1.2, 1.4, 1.3, 1.5, 1.6] * 2) 
HARDCODED_ACTIONS = np.array([1.5, 1.7, 1.6, 1.8, 1.9] * 2)  

def run_rmhc(gens, show=True):
    os.mkdir(EXPER_DIR)

    iters = NUM_ITERS
    genome = np.random.rand(NUM_ACTUATORS * 2)

    fitness_progress = []
    generation_times = []

    best_fitness, _, _ = run_simulation(iters, genome, show)
    print(f"Starting fitness: {best_fitness:.6f}")

    no_improvement_count = 0  

    for i in range(gens):
        mutated_genome = genome.copy()
        mutated_genome = np.array([
            random.random() if random.random() < MUTATE_RATE else x
            for x in mutated_genome
        ])

        new_fitness, avg_velocity, latency = run_simulation(iters, mutated_genome, show)

        fitness_progress.append(new_fitness)
        generation_times.append(i)

        if new_fitness > best_fitness:
            print(f"🟢 Found better fitness after {i} generations: {new_fitness:.6f}")
            best_fitness = new_fitness
            genome = mutated_genome
            no_improvement_count = 0  
        else:
            no_improvement_count += 1

        if latency:
            print(f"⚡ Latency for generation {i}: {latency} simulation steps")

        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            print(f"🛑 Stopping early: No fitness improvement for {NO_IMPROVEMENT_LIMIT} generations")
            break

        if avg_velocity < STOPPING_THRESHOLD:
            print(f"🛑 Stopping early: Robot velocity is too low ({avg_velocity:.6f})")
            break

    print(f"\nFinal fitness after {i+1} generations: {best_fitness:.6f}")

    log_fitness_progress(fitness_progress, generation_times)
    run_simulation(NUM_ITERS * 5, genome, fittest=True)

    return genome, best_fitness

def run_simulation(iters, genome, show=True, fittest=False):
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

    latency_start = None
    latency_end = None
    latency_triggered = False
    first_phase_captured = False  

    corners = find_corners(ROBOT_FILE_NAME)

    for step in range(iters):
        pos_1 = sim.object_pos_at_time(sim.get_time(), "robot")
        com_1 = np.mean(pos_1, 1)

        if latency_triggered:
            action = HARDCODED_ACTIONS
        else:
            action = genome * ((com_1[0] + com_1[1]) / 2)

        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        action = action[:NUM_ACTUATORS] if step % (FRAME_CYCLE_LEN * 2) < FRAME_CYCLE_LEN else action[NUM_ACTUATORS:]
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

            if step > 20 and velocity < STOPPING_THRESHOLD and latency_start is None and not first_phase_captured:
                first_phase_captured = True
                print(f"🟡 Initial fall impact detected at step {step}")

            if first_phase_captured and velocity < STOPPING_THRESHOLD and latency_start is None:
                latency_start = step  
                print(f"🔴 First stop detected at step {step}")
            elif velocity > STOPPING_THRESHOLD and latency_start is not None and not latency_triggered:
                latency_triggered = True  
                print(f"🟢 Robot restarted movement at step {step}")
            elif velocity < STOPPING_THRESHOLD and latency_triggered and latency_end is None:
                latency_end = step  
                print(f"🔴 Second stop detected at step {step}")
                break  

        prev_com = com_2

        if show:
            viewer.render('screen', verbose=False)

    viewer.close()
    avg_velocity = total_velocity / iters

    latency = latency_end - latency_start if latency_start and latency_end else None

    log_velocity_time(velocity_over_time)
    return fitness, avg_velocity, latency


def log_velocity_time(velocity_over_time):
    """Save velocity progression over simulation steps."""
    df = pd.DataFrame({'Step': range(len(velocity_over_time)), 'Velocity': velocity_over_time})
    df.to_csv(os.path.join(EXPER_DIR, "velocity_time_log.csv"), index=False)
    plt.plot(range(len(velocity_over_time)), velocity_over_time, label="Velocity Over Steps")
    plt.xlabel("Simulation Step")
    plt.ylabel("Velocity")
    plt.legend()
    plt.savefig(os.path.join(EXPER_DIR, "velocity_time_plot.png"))
    plt.close()


def log_fitness_progress(fitness_progress, generation_times):
    """Save fitness progress over generations."""
    df = pd.DataFrame({'Generation': generation_times, 'Fitness': fitness_progress})
    df.to_csv(os.path.join(EXPER_DIR, "fitness_progress_log.csv"), index=False)
    plt.plot(generation_times, fitness_progress, label="Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig(os.path.join(EXPER_DIR, "fitness_progress_plot.png"))
    plt.close()


if __name__ == "__main__":
    run_rmhc(GENS, False)