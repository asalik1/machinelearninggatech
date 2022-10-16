import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# Format the dataset
# Get dataset
df = pd.read_csv("titanic.csv")
# Drop unneeded column(s)
df = df.drop('Name', axis=1)
df.dropna()
# Use label encoder to reclassify 'sex' category
le = LabelEncoder()
df.Sex = le.fit_transform(df.Sex)
# x will be all the columns except the 'survive' column
x = df.iloc[:,1:]
# y will be the survive column
y = df.iloc[:,0]
# Split dataset into 0.8 training and 0.2 testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# Normalize feature data
scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Initial Parameters")
lr_model1 = mlrose.NeuralNetwork(hidden_nodes = [128], activation = 'relu', 
                                 algorithm = 'genetic_alg', 
                                 max_iters = 200, bias = True, is_classifier = True, 
                                 learning_rate = 0.01, early_stopping = True, 
                                 clip_max=5, max_attempts = 100, random_state = None,
                                 pop_size=800, mutation_prob=0.01)

lr_model1.fit(x_train_scaled, y_train)
print("LOSS: ", lr_model1.loss)
# Predict labels for train set and assess accuracy
y_train_pred = lr_model1.predict(x_train_scaled)
y_test_pred = lr_model1.predict(x_test_scaled)
print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
# scores = cross_val_score(lr_model1, x_train, y_train, cv=5)
# print ( 'On average, this model is correct {:0.2f}% (+/- {:0.2f}%) of the time.'.format(
#         scores.mean() * 100.0, scores.std() * 2 * 100.0))

#-------------------------------------------------------------
# Optimization Algorithm Analysis
# fitted_weights (array) –- Numpy array giving the fitted weights when fit is performed.
# loss (float) – Value of loss function for fitted weights when fit is performed.
# predicted_probs (array) –- Numpy array giving the predicted probabilities for each class 
# when predict is performed for multi-class classification data; or the predicted probability 
# for class 1 when predict is performed for binary classification data.
# fitness_curve (array) –- Numpy array giving the fitness at each training iteration.

# print("One Max Fitness Score:")
# # length = len(lr_model1.predicted_probs)
# length = 178
# fitness_fn = mlrose.OneMax()
# problem  = mlrose.DiscreteOpt(length, fitness_fn, maximize=True, max_val=2)

# best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=800, mutation_prob=0.1, max_attempts=100, 
#                                                             max_iters= 200, curve=True, random_state=None)
# print(best_fitness)

# print("Four Peaks Fitness Score:")
# fitness_fn = mlrose.FourPeaks(t_pct=0.1)
# best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=800, mutation_prob=0.1, max_attempts=100, 
#                                                             max_iters= 200, curve=True, random_state=None)
# # print(best_state)
# print(best_fitness)
# # print(fitness_curve)

# print("Six Peaks Fitness Score:")
# fitness_fn = mlrose.SixPeaks(t_pct=0.1)
# best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=800, mutation_prob=0.1, max_attempts=100,
#                                                             max_iters= 200, curve=True, random_state=None)
# # print(best_state)
# print(best_fitness)
# # print(fitness_curve)

# fitness_fn = mlrose.SixPeaks(t_pct=0.1)
# problem  = mlrose.DiscreteOpt(length, fitness_fn, maximize=True, max_val=2)
# runner_return = mlrose.GARunner(problem=problem,
#                           experiment_name="GA_Exp",
#                           seed=50,
#                           iteration_list=[1000],
#                           max_attempts=100,
#                           population_sizes=[50, 200, 500],
#                           mutation_rates=[0.1, 0.25, 0.5])

# ga_run_stats, ga_run_curves = runner_return.run()

# best_index_in_curve = ga_run_curves.Fitness.idxmax()
# last_iters = ga_run_stats[ga_run_stats.Iteration != 0].reset_index()
# print("Max and mean")
# print(last_iters.Fitness.max(), last_iters.Fitness.mean(), last_iters.Time.mean())
# print(last_iters.groupby("Mutation Rate").Fitness.mean())
# print(last_iters.groupby("Population Size").Fitness.mean())
# print(last_iters.groupby("Population Size").Time.mean())

#-------------------------------------------------------------
# Generating Plots

# # Accuracy Score vs Max Iterations
# # Setting the range for the parameter
# parameter_range = np.arange(100, 300, 100)
 
# train_score, test_score = validation_curve(lr_model1, x, y,
#                                        param_name = "max_iters",
#                                        param_range = parameter_range,
#                                         cv = 5, scoring = "accuracy")
 
# # Calculating mean and standard deviation of training score
# mean_train_score = np.mean(train_score, axis = 1)
# std_train_score = np.std(train_score, axis = 1)
 
# # Calculating mean and standard deviation of testing score
# mean_test_score = np.mean(test_score, axis = 1)
# std_test_score = np.std(test_score, axis = 1)
 
# # Plot mean accuracy scores for training and testing scores
# plt.plot(parameter_range, mean_train_score,
#      label = "Training Score", color = 'b')
# plt.plot(parameter_range, mean_test_score,
#    label = "CV Score", color = 'g')
 
# # Creating the plot
# plt.title("Validation Curve")
# plt.xlabel("Max Iter")
# plt.ylabel("Accuracy")
# plt.tight_layout()
# plt.legend(loc = 'best')
# plt.show()

# Accuracy Score vs Learning Rate
# Setting the range for the parameter
# parameter_range = np.arange(0.1, 1, 0.1)
 
# train_score, test_score = validation_curve(lr_model1, x, y,
#                                        param_name = "learning_rate",
#                                        param_range = parameter_range,
#                                         cv = 5, scoring = "accuracy")
 
# # Calculating mean and standard deviation of training score
# mean_train_score = np.mean(train_score, axis = 1)
# std_train_score = np.std(train_score, axis = 1)
 
# # Calculating mean and standard deviation of testing score
# mean_test_score = np.mean(test_score, axis = 1)
# std_test_score = np.std(test_score, axis = 1)
 
# # Plot mean accuracy scores for training and testing scores
# plt.plot(parameter_range, mean_train_score,
#      label = "Training Score", color = 'b')
# plt.plot(parameter_range, mean_test_score,
#    label = "CV Score", color = 'g')
 
# # Creating the plot
# plt.title("Validation Curve")
# plt.xlabel("Learning Rate")
# plt.ylabel("Accuracy")
# plt.tight_layout()
# plt.legend(loc = 'best')
# plt.show()

# # Accuracy Score vs Pop Size
# # Setting the range for the parameter
# parameter_range = np.arange(50, 200, 50)
 
# train_score, test_score = validation_curve(lr_model1, x, y,
#                                        param_name = "pop_size",
#                                        param_range = parameter_range,
#                                         cv = 5, scoring = "accuracy")
 
# # Calculating mean and standard deviation of training score
# mean_train_score = np.mean(train_score, axis = 1)
# std_train_score = np.std(train_score, axis = 1)
 
# # Calculating mean and standard deviation of testing score
# mean_test_score = np.mean(test_score, axis = 1)
# std_test_score = np.std(test_score, axis = 1)
 
# # Plot mean accuracy scores for training and testing scores
# plt.plot(parameter_range, mean_train_score,
#      label = "Training Score", color = 'b')
# plt.plot(parameter_range, mean_test_score,
#    label = "CV Score", color = 'g')
 
# # Creating the plot
# plt.title("Validation Curve")
# plt.xlabel("Pop Size")
# plt.ylabel("Accuracy")
# plt.tight_layout()
# plt.legend(loc = 'best')
# plt.show()

# # Accuracy Score vs Mutation Probability
# # Setting the range for the parameter
# parameter_range = np.arange(0.01, 1, 0.1)
 
# train_score, test_score = validation_curve(lr_model1, x, y,
#                                        param_name = "mutation_prob",
#                                        param_range = parameter_range,
#                                         cv = 5, scoring = "accuracy")
 
# # Calculating mean and standard deviation of training score
# mean_train_score = np.mean(train_score, axis = 1)
# std_train_score = np.std(train_score, axis = 1)
 
# # Calculating mean and standard deviation of testing score
# mean_test_score = np.mean(test_score, axis = 1)
# std_test_score = np.std(test_score, axis = 1)
 
# # Plot mean accuracy scores for training and testing scores
# plt.plot(parameter_range, mean_train_score,
#      label = "Training Score", color = 'b')
# plt.plot(parameter_range, mean_test_score,
#    label = "CV Score", color = 'g')
 
# # Creating the plot
# plt.title("Validation Curve")
# plt.xlabel("Mutation probability")
# plt.ylabel("Accuracy")
# plt.tight_layout()
# plt.legend(loc = 'best')
# plt.show()

# Learning Curve Plot
# def plot_learning_curve(
#     estimator,
#     title,
#     X,
#     y,
#     axes=None,
#     ylim=None,
#     cv=None,
#     n_jobs=None,
#     scoring=None,
#     train_sizes=np.linspace(0.1, 1.0, 5),
# ):
#     """
#     Generate 3 plots: the test and training learning curve, the training
#     samples vs fit times curve, the fit times vs score curve.

#     Parameters
#     ----------
#     estimator : estimator instance
#         An estimator instance implementing `fit` and `predict` methods which
#         will be cloned for each validation.

#     title : str
#         Title for the chart.

#     X : array-like of shape (n_samples, n_features)
#         Training vector, where ``n_samples`` is the number of samples and
#         ``n_features`` is the number of features.

#     y : array-like of shape (n_samples) or (n_samples, n_features)
#         Target relative to ``X`` for classification or regression;
#         None for unsupervised learning.

#     axes : array-like of shape (3,), default=None
#         Axes to use for plotting the curves.

#     ylim : tuple of shape (2,), default=None
#         Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

#     cv : int, cross-validation generator or an iterable, default=None
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:

#           - None, to use the default 5-fold cross-validation,
#           - integer, to specify the number of folds.
#           - :term:`CV splitter`,
#           - An iterable yielding (train, test) splits as arrays of indices.

#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.

#     n_jobs : int or None, default=None
#         Number of jobs to run in parallel.
#         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
#         for more details.

#     scoring : str or callable, default=None
#         A str (see model evaluation documentation) or
#         a scorer callable object / function with signature
#         ``scorer(estimator, X, y)``.

#     train_sizes : array-like of shape (n_ticks,)
#         Relative or absolute numbers of training examples that will be used to
#         generate the learning curve. If the ``dtype`` is float, it is regarded
#         as a fraction of the maximum size of the training set (that is
#         determined by the selected validation method), i.e. it has to be within
#         (0, 1]. Otherwise it is interpreted as absolute sizes of the training
#         sets. Note that for classification the number of samples usually have
#         to be big enough to contain at least one sample from each class.
#         (default: np.linspace(0.1, 1.0, 5))
#     """
#     if axes is None:
#         _, axes = plt.subplots(1, 3, figsize=(20, 5))

#     axes[0].set_title(title)
#     if ylim is not None:
#         axes[0].set_ylim(*ylim)
#     axes[0].set_xlabel("Training examples")
#     axes[0].set_ylabel("Score")

#     train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
#         estimator,
#         X,
#         y,
#         scoring=scoring,
#         cv=cv,
#         n_jobs=n_jobs,
#         train_sizes=train_sizes,
#         return_times=True,
#     )
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     # fit_times_mean = np.mean(fit_times, axis=1)
#     # fit_times_std = np.std(fit_times, axis=1)

#     # Plot learning curve
#     axes[0].grid()
#     axes[0].fill_between(
#         train_sizes,
#         train_scores_mean - train_scores_std,
#         train_scores_mean + train_scores_std,
#         alpha=0.1,
#         color="r",
#     )
#     axes[0].fill_between(
#         train_sizes,
#         test_scores_mean - test_scores_std,
#         test_scores_mean + test_scores_std,
#         alpha=0.1,
#         color="g",
#     )
#     axes[0].plot(
#         train_sizes, train_scores_mean, "o-", color="r", label="Training score"
#     )
#     axes[0].plot(
#         train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
#     )
#     axes[0].legend(loc="best")

#     # Plot n_samples vs fit_times
#     # axes[1].grid()
#     # axes[1].plot(train_sizes, fit_times_mean, "o-")
#     # axes[1].fill_between(
#     #     train_sizes,
#     #     fit_times_mean - fit_times_std,
#     #     fit_times_mean + fit_times_std,
#     #     alpha=0.1,
#     # )
#     # axes[1].set_xlabel("Training examples")
#     # axes[1].set_ylabel("fit_times")
#     # axes[1].set_title("Scalability of the model")

#     # Plot fit_time vs score
#     # fit_time_argsort = fit_times_mean.argsort()
#     # fit_time_sorted = fit_times_mean[fit_time_argsort]
#     # test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
#     # test_scores_std_sorted = test_scores_std[fit_time_argsort]
#     # axes[2].grid()
#     # axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
#     # axes[2].fill_between(
#     #     fit_time_sorted,
#     #     test_scores_mean_sorted - test_scores_std_sorted,
#     #     test_scores_mean_sorted + test_scores_std_sorted,
#     #     alpha=0.1,
#     # )
#     # axes[2].set_xlabel("fit_times")
#     # axes[2].set_ylabel("Score")
#     # axes[2].set_title("Performance of the model")

#     return plt


# #10,, 15
# fig, axes = plt.subplots(3, 2, figsize=(25, 15))

# title = "Learning Curves (Naive Bayes)"
# # Cross validation with 50 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

# estimator = lr_model1
# plt = plot_learning_curve(
#     estimator,
#     title,
#     x,
#     y,
#     axes=axes[:, 0],
#     ylim=None,
#     cv=5,
#     n_jobs=1,
#     scoring="accuracy",
# )

# plt.show()


