# Starting Framework

## `run_cma_es.py`

Runs cma-es along the lines of the `basic_use_case` demo, except this time implementing a sin-wave based actuator controller
rather than a closed-loop control scheme. Saves output as a csv file in ./out, which includes each individual's genome & fitness.
Number of generations and sigma can be passed as command line arguments Example: `python3 run.py 50 2` runs cma-es for 50 generations 
and a sigma of 2.

## `run_individual.py`

Run a single individual from its genome in an output csv file. Takes three command line args: filename, generation num, individual num.
Example: `python3 run_individual.py run_1738640858.csv 4 13`

## `plot.py`

Plots genome values on the x and y axis and color based on fitness. Takes three command line arguments: csv filename, x-axis name, y-axis name.
This is intended to show the relationship between two genome values and how it affects fitness.
Example: `python3 plot.py run_1738640858.csv frequency0 amplitude0`

## `plot_gens.py`

Plot two genome values with color-based fitness & generation slider. Takes three command line arguments: csv filename, x-axis name, y-axis name.
This is intended to demonstrate how cma-es discovers the search space over time.
Example: `python3 plot_gens.py run_1738640858.csv frequency0 amplitude0`

