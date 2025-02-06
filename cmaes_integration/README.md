# exp-framework

Implementation of an experimental framework for evolving SNN-controlled robots in evogym.

`python3 run_cmaes.py` runs cma-es with an open loop controlled robot. Creates output.csv and updates it continuously with the best individual from each generation. Number of generations and sigma can be passed as command line arguments.Example: `python3 run.py 50 2` runs cma-es for 50 generations and a sigma of 2.

Once `run_cmaes.py` is running, `python3 best_individual.py` can be run simultaneously, which plays the best found individual so far in a loop.
