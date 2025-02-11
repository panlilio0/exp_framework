# exp-framework

Implementation of an experimental framework for evolving SNN-controlled robots in evogym.

`python3 run_cmaes.py` runs cma-es with an open loop controlled robot. Creates output.csv and updates it continuously with the best individual from each generation. Whether to show the simulation or save as video, number of generations, sigma can be passed as command line arguments. Example: `python3 run_cmaes.py headless 50 2` runs cma-es for 50 generations in headless mode with a sigma of 2. Replacing "headless" with "screen" makes the simulation output to the screen, and replacing it with "video" saves each simulation as a video in `./videos` "both" shows on screen and saves a video.

Once `run_cmaes.py` is running, `python3 best_individual.py` can be run simultaneously, which plays the best found individual so far in a loop.
