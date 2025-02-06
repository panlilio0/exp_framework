import sys, pandas, run_simulation


ITERS = 200

"""
Look at output.csv and continuously run best individual.
Assumes csv names are their best achieved fitnesses
Continually searches for the lowest best fitness, plays the visualization and repeats
"""

def visualize_best(filename):

    df = pandas.read_csv("output.csv")

    try: #May need this in case file is currently being written to by run_cmaes

        best_fitness = min(df["Best-Fitness"])
        row = df.loc[df['Best-Fitness']==best_fitness]
        genome = row.values.tolist()[0][3:]
        run_simulation.run(genome, show=True)

    except:
        visualize_best(filename)

    visualize_best(filename)
    


if __name__ == "__main__":

    visualize_best("output.csv")

    