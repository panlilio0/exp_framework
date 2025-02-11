# exp-framework

Implementation of an experimental framework for evolving SNN-controlled robots in evogym.

## `run_cmaes.py`

Runs cma-es with an open loop controlled robot. Creates output.csv and updates it continuously with the best individual from each generation. Whether to show the simulation or save as video, number of generations, sigma can be passed as command line arguments. Example: `python3 run_cmaes.py headless 50 2` runs cma-es for 50 generations in headless mode with a sigma of 2. Replacing "headless" with "screen" makes the simulation output to the screen, and replacing it with "video" saves each simulation as a video in `./videos` "both" shows on screen and saves a video.

## `best_individual.py`

Once `run_cmaes.py` is running, `python3 best_individual.py` can be run simultaneously, which plays the best found individual so far in a loop.

## `run_individual.py` 

Runs a single individual from its genome in an output csv file. Takes one command line arg corresponding to generation number. Second command line argument tells whether to show simulation, save it to video, or both. "screen" renders the video to the screen. "video" saves a video to the "./videos" folder. "both" does both of these things. Example: `python3 run_individual.py 10 screen`

## `plot_genome.py`

Plots genome values on the x and y axis and color based on fitness. Takes three command line arguments: csv filename, x-axis name, y-axis name. This is intended to show the relationship between two genome values and how it affects fitness.
Example: `python3 plot.py output.csv frequency0 amplitude0`

## `plot_fitness_over_gens.py`

Given `output.csv`, plot fitness over generations.
