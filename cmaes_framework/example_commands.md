# Example commands

## Run experiment

Run experiment, continuously showing best individual so far in a window.
`python3 run_experiment.py --gens 50 --sigma 2 --mode s`

Run experiment, outputing the best individual from each generation as a video.
`python3 run_experiment.py --gens 50 --sigma 2 --mode v`

Run experiment, operating in headless mode
`python3 run_experiment.py --gens 50 --sigma 2 --mode h`

## Run cmaes

Run cmaes, outputing simulation to a window
`python3 run_cmaes.py --gens 50 --sigma 2 --mode s`

Run cmaes, outputing simulation to a video
`python3 run_cmaes.py --gens 50 --sigma 2 --mode v`

Run cmaes, operating in headless mode
`python3 run_cmaes.py --gens 50 --sigma 2 --mode h`

## Continually show best individual 
With two terminals open, run the first command. This should start running cmaes in headless mode
and continuosly updating output.csv with the best individual from each generation
`python3 run_cmaes.py --gens 50 --sigma 2 --mode h`
Now, in the second terminal, run the second command. This should start playing the most fit
individual currently found in output.csv, in a loop forever
`python3 best_individual.py --file 2025-02-24_21:15:03.csv`

## Show one individual
Assuming output.csv exists (generated from running run_cmaes.py)
show the best individual from generation 2 in a window 
`python3 run_individual.py --mode s --gen 2 --filename 2025-02-24_21:15:03.csv`
or save to a video
`python3 run_individual.py --mode v --gen 2 --filename 2025-02-24_21:15:03.csv`

## Plots
Plot frequency0 vs amplitude0 (two elements of the genome) and coloring based on fitness
`python3 plot_genome.py --filename 2025-02-24_21:15:03.csv --xaxis weight0 --yaxis weight1`

Plot best fitness per generation
`python3 plot_fitness_over_gens.py --filename 2025-02-24_21:15:03.csv`
