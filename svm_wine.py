import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV

# Boosting
# Get dataset
df = pd.read_csv("winequality-red.csv")
# Drop unneeded column(s)
df.dropna()
# x will be all the columns except the 'quality' column
x = df.iloc[:,:11]
# y will be the quality column
y = df.iloc[:,11]
# Split dataset into 0.8 training and 0.2 testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# Default rbf kernel
print("Default rbf kernel")
classifier = SVC()
classifier.fit(x_train, y_train)
y_train_pred = classifier.predict(x_train)
y_test_pred = classifier.predict(x_test)
print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
scores = cross_val_score(classifier, x_train, y_train, cv=5)
print ( 'On average, this model is correct {:0.2f}% (+/- {:0.2f}%) of the time.'.format(
        scores.mean() * 100.0, scores.std() * 2 * 100.0))

# Linear kernel
print("Linear Kernel")
classifier = SVC(kernel='linear')
classifier.fit(x_train, y_train)
y_train_pred = classifier.predict(x_train)
y_test_pred = classifier.predict(x_test)
print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
scores = cross_val_score(classifier, x_train, y_train, cv=5)
print ( 'On average, this model is correct {:0.2f}% (+/- {:0.2f}%) of the time.'.format(
        scores.mean() * 100.0, scores.std() * 2 * 100.0))


# Poly kernel
print("Poly Kernel")
classifier = SVC(kernel='poly')
classifier.fit(x_train, y_train)
y_train_pred = classifier.predict(x_train)
y_test_pred = classifier.predict(x_test)
print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
scores = cross_val_score(classifier, x_train, y_train, cv=5)
print ( 'On average, this model is correct {:0.2f}% (+/- {:0.2f}%) of the time.'.format(
        scores.mean() * 100.0, scores.std() * 2 * 100.0))

# Sigmoid kernel
print("Sigmoid Kernel")
classifier = SVC(kernel='sigmoid')
classifier.fit(x_train, y_train)
y_train_pred = classifier.predict(x_train)
y_test_pred = classifier.predict(x_test)
print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
scores = cross_val_score(classifier, x_train, y_train, cv=5)
print ( 'On average, this model is correct {:0.2f}% (+/- {:0.2f}%) of the time.'.format(
        scores.mean() * 100.0, scores.std() * 2 * 100.0))

print("Optimal Non-Linear Parameters")
param_grid = {'C': [0.1, 1, 10, 100],'kernel': ['rbf', 'poly', 'sigmoid'], 'coef0' : [0,1,2,3,4,5]}
grid = GridSearchCV(SVC(),param_grid,refit=True)
grid.fit(x_train,y_train)
print(grid.best_estimator_)
y_train_pred = grid.best_estimator_.predict(x_train)
y_test_pred = grid.best_estimator_.predict(x_test)
print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
scores = cross_val_score(grid, x_train, y_train, cv=5)
print ( 'On average, this model is correct {:0.2f}% (+/- {:0.2f}%) of the time.'.format(
        scores.mean() * 100.0, scores.std() * 2 * 100.0))

import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, "o-")
    # axes[1].fill_between(
    #     train_sizes,
    #     fit_times_mean - fit_times_std,
    #     fit_times_mean + fit_times_std,
    #     alpha=0.1,
    # )
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    # fit_time_argsort = fit_times_mean.argsort()
    # fit_time_sorted = fit_times_mean[fit_time_argsort]
    # test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    # test_scores_std_sorted = test_scores_std[fit_time_argsort]
    # axes[2].grid()
    # axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    # axes[2].fill_between(
    #     fit_time_sorted,
    #     test_scores_mean_sorted - test_scores_std_sorted,
    #     test_scores_mean_sorted + test_scores_std_sorted,
    #     alpha=0.1,
    # )
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt


#10,, 15
fig, axes = plt.subplots(3, 2, figsize=(25, 15))

title = "Learning Curves (Naive Bayes)"
# Cross validation with 50 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

estimator = SVC(kernel='poly', coef=3, C=10)
plt = plot_learning_curve(
    estimator,
    title,
    x,
    y,
    axes=axes[:, 0],
    ylim=None,
    cv=cv,
    n_jobs=1,
    scoring="accuracy",
)

plt.show()

from sklearn.model_selection import validation_curve
 
# Calculate accuracy on training and test set using the
# gamma parameter with 5-fold cross validation
parameter_range = np.arange(1,50,5)
 
# Calculate accuracy on training and test set using the
# gamma parameter with 5-fold cross validation
train_score, test_score = validation_curve(SVC(kernel='sigmoid'), x, y,
                                       param_name = "C",
                                       param_range = parameter_range,
                                        cv = 5, scoring = "accuracy")
 
# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis = 1)
std_train_score = np.std(train_score, axis = 1)
 
# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis = 1)
std_test_score = np.std(test_score, axis = 1)
 
# Plot mean accuracy scores for training and testing scores
plt.plot(parameter_range, mean_train_score,
    label = "Training Score (Sig)", color = 'b')
plt.plot(parameter_range, mean_test_score,
    label = "Cross Validation Score (Sig)", color = 'g')

# Calculate accuracy on training and test set using the
# gamma parameter with 5-fold cross validation
train_score, test_score = validation_curve(SVC(kernel='poly'), x, y,
                                       param_name = "C",
                                       param_range = parameter_range,
                                        cv = 5, scoring = "accuracy")
 
# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis = 1)
std_train_score = np.std(train_score, axis = 1)
 
# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis = 1)
std_test_score = np.std(test_score, axis = 1)

plt.plot(parameter_range, mean_train_score,
    label = "Training Score (Poly)", color = 'r')
plt.plot(parameter_range, mean_test_score,
    label = "Cross Validation Score (Poly)", color = 'm')

# # Calculate accuracy on training and test set using the
# # gamma parameter with 5-fold cross validation
# train_score, test_score = validation_curve(SVC(kernel='linear'), x, y,
#                                        param_name = "C",
#                                        param_range = parameter_range,
#                                         cv = 5, scoring = "accuracy")
 
# # Calculating mean and standard deviation of training score
# mean_train_score = np.mean(train_score, axis = 1)
# std_train_score = np.std(train_score, axis = 1)
 
# # Calculating mean and standard deviation of testing score
# mean_test_score = np.mean(test_score, axis = 1)
# std_test_score = np.std(test_score, axis = 1)

# plt.plot(parameter_range, mean_train_score,
#     label = "Training Score (Linear)", color = 'c')
# plt.plot(parameter_range, mean_test_score,
#     label = "Cross Validation Score (Linear)", color = 'y')

# Calculate accuracy on training and test set using the
# gamma parameter with 5-fold cross validation
train_score, test_score = validation_curve(SVC(kernel='rbf'), x, y,
                                       param_name = "C",
                                       param_range = parameter_range,
                                        cv = 5, scoring = "accuracy")
 
# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis = 1)
std_train_score = np.std(train_score, axis = 1)
 
# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis = 1)
std_test_score = np.std(test_score, axis = 1)

plt.plot(parameter_range, mean_train_score,
    label = "Training Score (rbf)", color = 'darkgreen')
plt.plot(parameter_range, mean_test_score,
    label = "Cross Validation Score (rbf)", color = 'orange')
 
# Creating the plot
plt.title("Validation Curve with SVC")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()