import mlrose_hiive as mlrose
import numpy as np

# Optimization Algorithm Analysis
# fitted_weights (array) –- Numpy array giving the fitted weights when fit is performed.
# loss (float) – Value of loss function for fitted weights when fit is performed.
# predicted_probs (array) –- Numpy array giving the predicted probabilities for each class 
# when predict is performed for multi-class classification data; or the predicted probability 
# for class 1 when predict is performed for binary classification data.
# fitness_curve (array) –- Numpy array giving the fitness at each training iteration.

length = 178
# print("One Max Fitness Score:")
# fitness_fn = mlrose.OneMax()
# problem  = mlrose.DiscreteOpt(length, fitness_fn, maximize=True, max_val=2)
# best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=10,
#                                                  curve=True, random_state=None)
# print(best_fitness)

# print("Four Peaks Fitness Score:")
# fitness_fn = mlrose.FourPeaks(t_pct=0.1)
# best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=10,
#                                                  curve=True, random_state=None)
# print(best_fitness)

# print("Six Peaks Fitness Score:")
# fitness_fn = mlrose.SixPeaks(t_pct=0.1)
# best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=10,
#                                                  curve=True, random_state=None)
# print(best_fitness)

print("-----KNAPSACK-----")
knapsack_len= 178
# weights between 10 to 40
knapsack_weights=np.random.uniform(0,4,knapsack_len)
# values between 20 to 30
knapsack_values=np.random.uniform(0,4,knapsack_len)
max_weight_pct = 0.5
problem = mlrose.KnapsackOpt(weights=knapsack_weights, values=knapsack_values,
                                  max_weight_pct=max_weight_pct)

best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=10,
                                                curve=True, random_state=None)

print(best_fitness)

mmc = mlrose.MIMICRunner(problem=problem,
                  experiment_name="MMC_Exp",
                  seed=44,
                  iteration_list=[100],
                  max_attempts=10,
                  population_sizes=[50, 200, 500],
                  keep_percent_list=[0.25, 0.5, 0.75],
                  use_fast_mimic=True)

# the two data frames will contain the results
mmc_run_stats, mmc_run_curves = mmc.run()

last_iters = mmc_run_stats[mmc_run_stats.Iteration != 0].reset_index()
print("Max and mean")
print(last_iters.Fitness.max(), last_iters.Fitness.mean(), last_iters.Time.mean())
print(last_iters.groupby("Keep Percent").Fitness.mean())
print(last_iters.groupby("Population Size").Fitness.mean())
print(last_iters.groupby("Population Size").Time.mean())

print("-----ONE MAX PEAK-----")
fitness_fn = mlrose.OneMax()
problem  = mlrose.DiscreteOpt(length, fitness_fn, maximize=True, max_val=2)

mmc = mlrose.MIMICRunner(problem=problem,
                  experiment_name="MMC_Exp",
                  seed=44,
                  iteration_list=[100],
                  max_attempts=10,
                  population_sizes=[50, 200, 500],
                  keep_percent_list=[0.25, 0.5, 0.75],
                  use_fast_mimic=True)

# the two data frames will contain the results
mmc_run_stats, mmc_run_curves = mmc.run()

last_iters = mmc_run_stats[mmc_run_stats.Iteration != 0].reset_index()
print("Max and mean")
print(last_iters.Fitness.max(), last_iters.Fitness.mean(), last_iters.Time.mean())
print(last_iters.groupby("Keep Percent").Fitness.mean())
print(last_iters.groupby("Population Size").Fitness.mean())
print(last_iters.groupby("Population Size").Time.mean())

print("-----SIX-PEAK-----")
fitness_fn = mlrose.SixPeaks(t_pct=0.1)
problem  = mlrose.DiscreteOpt(length, fitness_fn, maximize=True, max_val=2)

mmc = mlrose.MIMICRunner(problem=problem,
                  experiment_name="MMC_Exp",
                  seed=44,
                  iteration_list=[100],
                  max_attempts=10,
                  population_sizes=[50, 200, 500],
                  keep_percent_list=[0.25, 0.5, 0.75],
                  use_fast_mimic=True)

# the two data frames will contain the results
mmc_run_stats, mmc_run_curves = mmc.run()

last_iters = mmc_run_stats[mmc_run_stats.Iteration != 0].reset_index()
print("Max and mean")
print(last_iters.Fitness.max(), last_iters.Fitness.mean(), last_iters.Time.mean())
print(last_iters.groupby("Keep Percent").Fitness.mean())
print(last_iters.groupby("Population Size").Fitness.mean())
print(last_iters.groupby("Population Size").Time.mean())

print("-----FOUR-PEAK-----")
fitness_fn = mlrose.FourPeaks(t_pct=0.1)
problem  = mlrose.DiscreteOpt(length, fitness_fn, maximize=True, max_val=2)

mmc = mlrose.MIMICRunner(problem=problem,
                  experiment_name="MMC_Exp",
                  seed=44,
                  iteration_list=[100],
                  max_attempts=10,
                  population_sizes=[50, 200, 500],
                  keep_percent_list=[0.25, 0.5, 0.75],
                  use_fast_mimic=True)

# the two data frames will contain the results
mmc_run_stats, mmc_run_curves = mmc.run()

last_iters = mmc_run_stats[mmc_run_stats.Iteration != 0].reset_index()
print("Max and mean")
print(last_iters.Fitness.max(), last_iters.Fitness.mean(), last_iters.Time.mean())
print(last_iters.groupby("Keep Percent").Fitness.mean())
print(last_iters.groupby("Population Size").Fitness.mean())
print(last_iters.groupby("Population Size").Time.mean())





