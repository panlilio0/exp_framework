# Basic Use Case

`run.py` runs a basic cma-es demo with a walker robot which implements a simple closed loop feedback control scheme. Number of generations and sigma can be passed as command line arguments Example: `python3 run.py 50 2` runs cma-es for 50 generations and a sigma of 2.

## CMA-ES overview

`optimizer = CMA(mean=np.ones(NUM_ACTUATORS * 2), sigma=sigma)` inits a cma object. `mean` is the "average" starting genome, from which new ones will be created via a normal distribution. `sigma` is the standard deviation of the normal distribution.

`x = optimizer.ask()` asks cma-es for a new genome based on the random distribution and its current understand of the solution space.

`value = run_simulation(NUM_ITERS, x, False)` runs a simulation and returns a fitness value.

`solutions.append((x, value))` informs cma-es about the fitness values of all the genomes it created in the current generation, allowing it to update its understanding of the solution space.

`all_solutions` is a multi-dimensional array. The top-level axis corresponds to the generation. Elements inside each generation are a tuple pair of (genome, fitness). They are sorted from best to worst fitness.
