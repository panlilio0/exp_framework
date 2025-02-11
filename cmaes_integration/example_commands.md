# Example commands

## Run cmaes

Run cmaes, outputing simulation to a window
`python3 run_cmaes.py screen 10 2`

Run cmaes, outputing simulation to a video
`python3 run_cmaes.py video 10 2`

Run cmaes, operating in headless mode
`python3 run_cmaes.py video 10 2`

## Continually show best individual 
With two terminals open, run the first command. This should start running cmaes in headless mode
and continuosly updating output.csv with the best individual from each generation
`python3 run_cmaes.py headless 50 2`
Now, in the second terminal, run the second command. This should start playing the most fit
individual currently found in output.csv, in a loop forever
`python3 best_individual.py`

## Show one individual
Assuming output.csv exists (generated from running run_cmaes.py)
show the best individual from generation 2 in a window 
`python3 run_individual.py 2 screen`
or save to a video
`python3 run_individual.py 2 video`

## Plots
Plot frequency0 vs amplitude0 (two elements of the genome) and coloring based on fitness
`python3 plot_genome.py output.csv frequency0 amplitude0`

Plot best fitness per generation
`python3 plot_fitness_over_gens.py `
