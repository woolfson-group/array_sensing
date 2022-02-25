

# Functions to perform ML analysis on parsed BADASS data.

import copy
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from collections import OrderedDict
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from matplotlib.colors import BASE_COLORS, CSS4_COLORS
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mlxtend.evaluate import combined_ftest_5x2cv
from scipy.stats import f
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_selection import (
    f_classif, mutual_info_classif, SelectKBest
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, GroupKFold, LeaveOneGroupOut, LeaveOneOut,
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from sklearn.utils.multiclass import unique_labels
from types import GeneratorType

sns.set()

if __name__ == 'subroutines.train':
    from subroutines.parse_array_data import DefData
    from subroutines.exceptions import AlgorithmError, create_generator
else:
    from array_sensing.subroutines.parse_array_data import DefData
    from array_sensing.subroutines.exceptions import (
        AlgorithmError, create_generator
    )


def make_separate_subclass_splits(subclasses, subclass_splits):
    """
    Constructs generator that splits an input dataset into train and test splits
    specified by the user

    Input
    ----------
    - subclasses: Numpy array (1D) of subclass values
    - subclass_splits: Numpy array (2D) of subclasses, each row lists the
    subclasses to be included in an individual split. Subclasses should not be
    repeated more than once.

    Output
    ----------
    - splits: Generator that creates train test splits, in which the subclasses
    in each row of subclass_splits form the test set in successive splits
    """

    if type(subclasses) != np.ndarray:
        raise TypeError(
            'Expect "subclasses" to be a (1D) array of subclass values'
        )
    if len(subclasses.shape) != 1:
        raise ValueError('Expect "subclasses" to be a 1D array')
    if pd.DataFrame(subclasses, dtype=object).isna().any(axis=None):
        raise ValueError('NaN value(s) detected in "subclasses" array')

    if type(subclass_splits) != np.ndarray:
        raise TypeError(
            'Expect "subclass_splits" to be a (2D) array of subclass values'
        )
    if pd.DataFrame(subclass_splits, dtype=object).isna().any(axis=None):
        raise ValueError('NaN value(s) detected in "subclass_splits" array')
    if pd.unique(subclass_splits.flatten()).size != subclass_splits.size:
        raise ValueError(
            'Repeated subclass labels detected in "subclass_splits"'
        )

    for val in subclasses:
        if not val in subclass_splits.flatten():
            raise ValueError(
                'Subclass {} is found in "subclasses" but not "subclass_splits"'
                ''.format(val)
            )
    for val in subclass_splits.flatten():
        if not val in subclasses:
            raise ValueError(
                'Subclass {} is found in "subclass_splits" but not "subclasses"'
                ''.format(val)
            )

    for i in range(subclass_splits.shape[0]):
        split = []
        subclass_row = subclass_splits[i]
        for subclass_1 in subclass_row:
            for j in range(subclasses.shape[0]):
                subclass_2 = subclasses[j]
                if subclass_1 == subclass_2:
                    split.append(j)
        split = np.array(sorted(split))

        yield split


def bootstrap_data(x, y, features, scale, test=False):
    """
    Generates bootstrapped repeat of input arrays (x and y)

    Input
    ----------
    - x: Numpy array of x values.
    - y: Numpy array of y values.
    - features: List of input features corresponding to the columns in x.
    - scale: Boolean, defines whether to scale the data (by subtracting the
    median and dividing by the IQR) before calculating feature importances.
    By default is set to True since scaling the data will affect the
    importance scores.
    - test: Boolean describing whether the function is being run during the
    program's unit tests - by default is set to False

    Output
    ----------
    temp_x: Bootstrapped x array
    temp_y: Bootstrapped y array
    """

    if type(x) != np.ndarray:
        raise TypeError(
            'Expect "x" to be a (2D) array of x values'
        )
    if type(y) != np.ndarray:
        raise TypeError(
            'Expect "y" to be a (1D) array of y values'
        )
    if len(y.shape) != 1:
        raise ValueError(
            'Expect "y" to be a 1D array of y values'
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            'Different numbers of rows in arrays "x" and "y"'
        )
    if type(features) != list:
        raise TypeError(
            'Expect "features" to be a list'
        )
    if len(features) != x.shape[1]:
        raise ValueError(
            'Expect entries in "features" list to correspond to the columns in "x"'
        )
    if type(scale) != bool:
        raise TypeError(
            'Expect "scale" to be a Boolean value (either True or False)'
        )
    test_rand_ints = [0, 6, 1, 3, 4, 5, 8, 0, 6, 5, 4, 7]

    random_rows = []
    temp_x = pd.DataFrame(
        data=copy.deepcopy(x), index=None, columns=features
    )
    temp_y = pd.DataFrame(
        data=copy.deepcopy(y), index=None, columns=['Analyte']
    )
    for m in range(temp_x.shape[0]):
        if test is False:
            random_rows.append(random.randint(0, (temp_x.shape[0]-1)))
        else:
            random_rows.append(test_rand_ints[m])
    temp_x = temp_x.iloc[random_rows,:].reset_index(drop=True)
    temp_y = temp_y.iloc[random_rows,:].reset_index(drop=True)['Analyte'].tolist()

    if scale is True:
        temp_x_scaled = RobustScaler().fit_transform(temp_x)
        temp_x = pd.DataFrame(temp_x_scaled, index=None, columns=features)

    return temp_x, temp_y


def make_feat_importance_plots(
    feature_importances, results_dir, plt_name, test
):
    """
    Generates plots for feature importance score functions

    Input
    ----------
    - feature_importances: Dictionary of importance scores for the different
    features included in the dataset
    - results_dir: Directory where the plots are to be saved
    - plt_name: Prefix to append to the names of the saved plots
    - test: Boolean describing whether the function is being run during the
    program's unit tests

    Output
    ----------
    - importance_df: DataFrame of features and their importance scores
    """

    if not type(feature_importances) in [dict, OrderedDict]:
        raise TypeError(
            'Expect "feature_importances" to be a dictionary of importance '
            'scores'
        )
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            'Directory {} does not exist'.format(results_dir)
        )
    if type(plt_name) != str:
        raise TypeError(
            'Expect "plt_name" to a string to append to the start of the names '
            'of the saved plots'
        )
    if os.path.isfile('{}/{}_feat_importance_percentiles.svg'.format(
        results_dir, plt_name
    )):
        raise FileExistsError(
            'File {}/{}_feat_importance_percentiles.svg already exists - please'
            ' rename this file so it is not overwritten by running this '
            'function'.format(results_dir, plt_name)
        )
    if os.path.isfile('{}/{}_feat_importance_all_data.svg'.format(
        results_dir, plt_name
    )):
        raise FileExistsError(
            'File {}/{}_feat_importance_all_data.svg already exists - please'
            ' rename this file so it is not overwritten by running this '
            'function'.format(results_dir, plt_name)
        )

    cols = []
    cols_all = []
    all_vals = []
    median_vals = []
    lower_conf_limit_vals = []
    upper_conf_limit_vals = []
    for col, importances in feature_importances.items():
        cols.append(col)
        median_vals.append(np.median(importances))
        lower_conf_limit_vals.append(np.percentile(importances, 2.5))
        upper_conf_limit_vals.append(np.percentile(importances, 97.5))
        for importance in importances:
            cols_all.append(col)
            all_vals.append(importance)

    plt.clf()
    plt.figure(figsize=(15,6))
    sns.barplot(x=cols, y=median_vals)
    sns.stripplot(x=cols, y=lower_conf_limit_vals, edgecolor='k',
                  linewidth=1, s=6, jitter=False)
    sns.stripplot(x=cols, y=upper_conf_limit_vals, edgecolor='k',
                  linewidth=1, s=6, jitter=False)
    plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
    plt.xlabel('Peptide')
    plt.ylabel('Importance')
    plt.savefig('{}/{}_feat_importance_percentiles.svg'.format(
        results_dir, plt_name
    ))
    if test is False:
        plt.show()

    plt.clf()
    plt.figure(figsize=(15,6))
    sns.stripplot(x=cols_all, y=all_vals, edgecolor='k', linewidth=1,
                  size=2.5, alpha=0.2)
    plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
    plt.xlabel('Peptide')
    plt.ylabel('Importance')
    plt.savefig('{}/{}_feat_importance_all_data.svg'.format(
        results_dir, plt_name
    ))
    if test is False:
        plt.show()

    importance_df = pd.DataFrame({
        'Feature': cols,
        'Score': median_vals,
        'Lower conf limit': lower_conf_limit_vals,
        'Upper conf limit': upper_conf_limit_vals
    })
    importance_df = importance_df.sort_values(
        by=['Score'], axis=0, ascending=False
    ).reset_index(drop=True)

    if test is False:
        return importance_df
    else:
        return (
            importance_df, cols, cols_all, all_vals, median_vals,
            lower_conf_limit_vals, upper_conf_limit_vals
        )


def draw_conf_matrices(y_test, predictions, results_dir, plt_name, test=False):
    """
    Draw confusion matrices for y_test vs. predictions

    Input
    ----------
    - y_test: Numpy array of real labels
    - predictions: Numpy array of predicted labels
    - results_dir: Directory where the plots are to be saved
    - plt_name: Prefix to append to the names of the saved plots
    - test: Boolean describing whether the function is being run during the
    program's unit tests - by default is set to False
    """

    if type(y_test) != np.ndarray:
        raise TypeError(
            'Expect "y_test" to be a (1D) numpy array of the real class labels '
            ' of the test dataset'
        )

    if type(predictions) != np.ndarray:
        raise TypeError(
            'Expect "predictions" to be a (1D) numpy array of the predicted '
            'class labels for the test dataset'
        )

    if len(y_test.shape) != 1:
        raise ValueError(
            'Expect "y_test" to be a (1D) numpy array of the real class labels '
            ' of the test dataset'
        )

    if len(predictions.shape) != 1:
        raise ValueError(
            'Expect "predictions" to be a (1D) numpy array of the predicted '
            'class labels for the test dataset'
        )

    if y_test.shape != predictions.shape:
        raise ValueError(
            'Mismatch in the number of classes in "y_test" and "predictions"'
        )

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            'Directory {} does not exist'.format(results_dir)
        )

    if type(plt_name) != str:
        raise TypeError(
            'Expect "plt_name" to a string to append to the start of the names '
            'of the saved plots'
        )

    if os.path.isfile('{}/{}_confusion_matrix.svg'.format(
        results_dir, plt_name
    )):
        raise FileExistsError(
            'File {}/{}_confusion_matrix.svg already exists - please rename '
            'this file so it is not overwritten by running this '
            'function'.format(results_dir, plt_name)
        )

    if os.path.isfile('{}/{}_recall_confusion_matrix.svg'.format(
        results_dir, plt_name
    )):
        raise FileExistsError(
            'File {}/{}_recall_confusion_matrix.svg already exists - please '
            'rename this file so it is not overwritten by running this '
            'function'.format(results_dir, plt_name)
        )

    if os.path.isfile('{}/{}_precision_confusion_matrix.svg'.format(
        results_dir, plt_name
    )):
        raise FileExistsError(
            'File {}/{}_precision_confusion_matrix.svg already exists - please '
            'rename this file so it is not overwritten by running this '
            'function'.format(results_dir, plt_name)
        )

    normalisation_methods = OrderedDict({None: [''],
                                         'true': ['_recall', 'rows'],
                                         'pred': ['_precision', 'columns']})
    for method, method_label in normalisation_methods.items():
        if not method is None:
            print('Normalised over {} label ({})'.format(
                method, method_label[1]
            ))

        plt.clf()
        labels = unique_labels(y_test, predictions)
        # Below ensures that predicted and true labels are on the
        # correct axes, so think carefully before updating!
        if method is None:
            sns.heatmap(
                data=confusion_matrix(y_true=y_test, y_pred=predictions,
                                      labels=labels, normalize=method),
                cmap='RdBu_r', annot=True, xticklabels=True,
                yticklabels=True, fmt='.3f'
            )
        else:
            sns.heatmap(
                data=confusion_matrix(y_true=y_test, y_pred=predictions,
                                      labels=labels, normalize=method),
                cmap='RdBu_r', annot=True, xticklabels=True,
                yticklabels=True, fmt='.3f', vmin=0, vmax=1
            )
        ax = plt.gca()
        ax.set(
            xticklabels=labels, yticklabels=labels, xlabel='Predicted label',
            ylabel='True label'
        )
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        plt.savefig('{}/{}{}_confusion_matrix.svg'.format(
            results_dir, plt_name, method_label[0]
        ))
        if test is False:
            plt.show()


def check_arguments(
    func_name, x_train, y_train, train_groups, x_test, y_test,
    selected_features, splits, const_split, resampling_method, n_components_pca,
    run, fixed_params, tuned_params, train_scoring_metric, test_scoring_funcs,
    n_iter, cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
    test=False
):
    """
    Tests whether the input arguments are of the expected type (and where
    appropriate equal to a value within a specified range)

    Input
    ----------
    - func_name: Name of the function "check_arguments" is being called within
    - x_train: Numpy array of x values of training data
    - y_train: Numpy array of y values of training data
    - train_groups: Numpy array of group names of training data
    - x_test: Numpy array of x values of test data
    - y_test: Numpy array of y values of test data
    - selected_features: List of features to include
    - splits: List of train: test splits of the input data (x_train and
    y_train) for cross-validation
    - const_split: Boolean, if set to True and splits is set to None,
    function will generate splits using a fixed random_state (=>
    reproducible results when the function is re-run)
    - resampling_method: Name of the method (string) used to resample the
    data in an imbalanced dataset. Recognised method names are:
    'no_balancing'; 'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
    - n_components_pca: The number of components to transform the data to after
    fitting the data with PCA. If set to None, PCA will not be included in
    the pipeline.
    - run: If func_name is "run_ml", one of "randomsearch", "gridsearch" or
    train". If func_name is "run_nested_CV", either "randomsearch" or
    "gridsearch".
    - fixed_params: Dictionary of fixed-value hyperparameters and their
    selected values, e.g. {n_jobs: -1}
    - tuned_params: Dictionary of hyperparameter values to search.
    Key = hyperparameter name (must match the name of the parameter in the
    selected sklearn ML classifier class); Value = range of values to
    test for that hyperparameter - note that all numerical ranges must be
    supplied as numpy arrays in order to avoid throwing an error with the
    imblearn Pipeline() class
    - train_scoring_metric: Name of the scoring metric used to measure the
    performance of the fitted classifier. Metric must be a string and be one of
    the scoring metrics for classifiers listed at
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
    - test_scoring_funcs: Dictionary of sklearn scoring functions and
    dictionaries of arguments to be fed into these functions,
    e.g. {sklearn.metrics.f1_score: {'average': 'macro'}}
    - n_iter: The number of hyperparameter combinations to test (either an
    integer value or None)
    - cv_folds_inner_loop: Integer number of folds to run in the inner
    cross-validation loop
    - cv_folds_outer_loop: Either an integer number of folds to run in the outer
    cross-validation loop, or 'loocv' (leave-one-out cross-validation, sets the
    number of folds equal to the size of the dataset)
    - draw_conf_mat: Boolean, dictates whether to plot confusion matrices to
    compare the model predictions to the test data.
    - plt_name: Prefix to append to name of the saved plot(s)
    - test: Boolean describing whether the function is being run during the
    program's unit tests - by default is set to False
    """

    # Tests that input data is provided as numpy arrays and that their
    # dimensions match up
    if type(x_train) != np.ndarray:
        raise TypeError(
            'Expect "x_train" to be a numpy array of training data fluorescence'
            ' readings'
        )
    else:
        if x_train.size > 0:
            x_train_cols = x_train.shape[1]
        else:
            x_train_cols = 0

    if type(y_train) != np.ndarray:
        raise TypeError(
            'Expect "y_train" to be a numpy array of training data class labels'
        )
    else:
        if y_train.size > 0:
            try:
                y_train.shape[1]
                raise ValueError('Expect "y_train" to be a 1D array')
            except IndexError:
                pass

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            'Different number of entries (rows) in "x_train" and "y_train"'
        )

    if not train_groups is None:
        if type(train_groups) != np.ndarray:
            raise TypeError(
                'Expect "train_groups" to be a numpy array of training data '
                'subclass labels'
            )
        else:
            if train_groups.size > 0:
                try:
                    train_groups.shape[1]
                    raise ValueError('Expect "train_groups" to be a 1D array')
                except IndexError:
                    pass
        if x_train.shape[0] != train_groups.shape[0]:
            raise ValueError(
                'Different number of entries (rows) in "x_train" and '
                '"train_groups"'
            )

    if type(x_test) != np.ndarray:
        raise TypeError(
            'Expect "x_test" to be a numpy array of test data fluorescence'
            ' readings'
        )
    else:
        if x_test.size > 0:
            x_test_cols = x_test.shape[1]
        else:
            x_test_cols = 0

    if type(y_test) != np.ndarray:
        raise TypeError(
            'Expect "y_test" to be a numpy array of test data class labels'
        )
    else:
        if y_test.size > 0:
            try:
                y_test.shape[1]
                raise ValueError('Expect "y_test" to be a 1D array')
            except IndexError:
                pass

    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            'Different number of entries (rows) in "x_test" and "y_test"'
        )

    if x_train_cols != 0 and x_test_cols != 0:
        if x_train_cols != x_test_cols:
            raise ValueError(
                'Different number of features incorporated in the training and '
                'test data'
            )

    if pd.DataFrame(x_train, dtype=object).isna().any(axis=None):
        raise ValueError('NaN value(s) detected in "x_train" data')
    if pd.DataFrame(y_train, dtype=object).isna().any(axis=None):
        raise ValueError('NaN value(s) detected in "y_train" data')
    if pd.DataFrame(train_groups, dtype=object).isna().any(axis=None):
        raise ValueError('NaN value(s) detected in "train_groups" data')
    if pd.DataFrame(x_test, dtype=object).isna().any(axis=None):
        raise ValueError('NaN value(s) detected in "x_test" data')
    if pd.DataFrame(y_test, dtype=object).isna().any(axis=None):
        raise ValueError('NaN value(s) detected in "y_test" data')

    if pd.DataFrame(x_train).applymap(
        lambda x: isinstance(x, (int, float))).all(axis=None, skipna=False
    ) is np.bool_(False):
        raise ValueError(
            'Non-numeric value(s) in "x_train" - expect all values in "x_train"'
            ' to be integers / floats'
        )
    if pd.DataFrame(x_test).applymap(
        lambda x: isinstance(x, (int, float))).all(axis=None, skipna=False
    ) is np.bool_(False):
        raise ValueError(
            'Non-numeric value(s) in "x_test" - expect all values in "x_test"'
            ' to be integers / floats'
        )

    # Tests arguments controlling the analysis of the input data
    if not type(selected_features) in [list, int]:
        raise TypeError(
            'Expect "selected_features" to be either a list of features to '
            'retain in the analysis, or an integer number of features (to be '
            'selected via permutation analysis)'
        )
    else:
        if type(selected_features) == list:
            len_selected_features = len(selected_features)
        else:
            len_selected_features = selected_features
            if len_selected_features < 1:
                raise ValueError(
                    'The number of selected_features must be a positive integer'
                )

        if x_train_cols != 0:
            if len_selected_features > x_train_cols:
                raise ValueError(
                    'There is a greater number of features in '
                    '"selected_features" than there are columns in the '
                    '"x_train" input arrays'
                )
        if x_test_cols != 0:
            if len_selected_features > x_test_cols:
                raise ValueError(
                    'There is a greater number of features in '
                    '"selected_features" than there are columns in the '
                    '"x_test" input arrays'
                )

    if type(splits) != list:
        raise TypeError(
            'Expect "splits" to be a list of train/test splits'
        )
    else:
        for split in splits:
            if (split[0].shape[0] + split[1].shape[0]) != x_train.shape[0]:
                raise ValueError(
                    'Size of train test splits generated by "splits" does not '
                    'match the number of rows in the input array "x_train"'
                )

    if type(const_split) != bool:
        raise TypeError(
            'Expect "const_split" to be a Boolean (True or False)'
        )

    exp_resampling_methods = [
        'no_balancing', 'max_sampling', 'smote', 'smoteenn', 'smotetomek'
    ]
    if not resampling_method in exp_resampling_methods:
        raise ValueError(
            '"resampling_method" unrecognised - expect value to be one of the '
            'following list entries:\n{}'.format(exp_resampling_methods)
        )

    if not n_components_pca is None:
        if type(n_components_pca) != int:
            raise TypeError(
                'Expect "n_components_pca" to be set either to None or to a '
                'positive integer value between 1 and the number of features'
            )
        else:
            if x_train_cols > 0:
                if n_components_pca < 1 or n_components_pca > x_train_cols:
                    raise ValueError(
                        'Expect "n_components_pca" to be set either to None or to '
                        'a positive integer value between 1 and the number of '
                        'features'
                    )
            else:
                if n_components_pca < 1 or n_components_pca > x_test_cols:
                    raise ValueError(
                        'Expect "n_components_pca" to be set either to None or to '
                        'a positive integer value between 1 and the number of '
                        'features'
                    )

    if func_name == 'run_ml':
        if not run in ['randomsearch', 'gridsearch', 'train']:
            raise ValueError(
                'Expect "run" to be set to either "randomsearch", "gridsearch" '
                'or "train"'
            )
    elif func_name == 'run_nested_CV':
        if not run in ['randomsearch', 'gridsearch']:
            raise ValueError(
                'Expect "run" to be set to either "randomsearch" or '
                '"gridsearch"'
            )

    if not type(fixed_params) in [dict, OrderedDict]:
        raise TypeError(
            'Expect "fixed_params" to be a dictionary of parameter values with '
            'which to run the selected classifier algorithm'
        )

    if not type(tuned_params) in [dict, OrderedDict]:
        raise TypeError(
            'Expect "tuned_params" to be a dictionary of parameter names (keys)'
            ' and ranges of values to optimise (values) using either random or '
            'grid search'
        )

    exp_train_score_metrics = [
        'accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision',
        'neg_brier_score', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted',
        'f1_samples', 'neg_log_loss', 'precision', 'precision_micro',
        'precision_macro', 'precision_weighted', 'precision_samples', 'recall',
        'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples',
        'jaccard', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted',
        'jaccard_samples', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
        'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'
    ]
    if type(train_scoring_metric) == sklearn.metrics._scorer._PredictScorer:
        pass
    else:
        if not train_scoring_metric in exp_train_score_metrics:
            raise ValueError(
                '"train_scoring_metric" not recogised - please specify a string'
                ' corresponding to the name of the metric you would like to use'
                ' in the sklearn.metrics module, e.g. "accuracy".\nExpect '
                'metric to be in the following list:\n'
                '{}'.format(exp_train_score_metrics)
            )

    exp_test_scoring_funcs = [
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, cohen_kappa_score
    ]
    for scoring_func, scoring_params in test_scoring_funcs.items():
        if not scoring_func in exp_test_scoring_funcs:
            raise ValueError(
                'Scoring function {} not recognised.\nExpect scoring functions '
                'to be in the following list:\n'
                '{}'.format(
                    scoring_func.__name__,
                    [scoring_func.__name__ for scoring_func in exp_test_scoring_funcs]
                )
            )
        if not type(scoring_params) in [dict, OrderedDict]:
            raise TypeError('Expect scoring parameters to be a dictionary')

    if not n_iter is None:
        if type(n_iter) != int:
            raise TypeError(
                '"n_iter" should be set to a positive integer value'
            )
        else:
            if n_iter < 1:
                raise ValueError(
                    '"n_iter" should be set to a positive integer value'
                )

    if type(cv_folds_inner_loop) != int:
        raise TypeError(
            'Expect "cv_folds_inner_loop" to be a positive integer value in the'
            ' range of 2 - 20'
        )
    else:
        if cv_folds_inner_loop < 2 or cv_folds_inner_loop > 20:
            raise ValueError(
                'Expect "cv_folds_inner_loop" to be a positive integer value in'
                ' the range of 2 - 20'
            )

    if type(cv_folds_outer_loop) == str:
        if cv_folds_outer_loop != 'loocv':
            raise ValueError(
                'Expect "cv_folds_outer_loop" to be set to either "loocv" '
                '(leave-one-out cross-validation) or a positive integer in the '
                'range of 2 - 20'
            )
    elif type(cv_folds_outer_loop) == int:
        if cv_folds_outer_loop < 2 or cv_folds_outer_loop > 20:
            raise ValueError(
                'Expect "cv_folds_outer_loop" to be set to either "loocv" '
                '(leave-one-out cross-validation) or a positive integer in the '
                'range of 2 - 20'
            )
    else:
        raise TypeError(
            'Expect "cv_folds_outer_loop" to be set to either "loocv" '
            '(leave-one-out cross-validation) or a positive integer in the '
            'range of 2 - 20'
        )

    if type(draw_conf_mat) != bool:
        raise TypeError(
            'Expect "draw_conf_mat" to be a Boolean value (True or False)'
        )

    if type(plt_name) != str:
        raise TypeError(
            'Expect "plt_name" to be a string'
        )

    if test is True:
        return 'All checks passed'


class ManualFeatureSelection(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, all_features, selected_features):
        """
        - all_features: A list/array of all features in the input data
        - selected_features: A list/array of the features to be retained
        """

        self.all_features = all_features
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Removes features not in "selected_features" from X
        """

        df = pd.DataFrame(data=X, index=None, columns=self.all_features)
        sub_df = df[self.selected_features]
        X = sub_df.to_numpy()

        return X


class RunML(DefData):

    def __init__(
        self, results_dir, fluor_data, classes=None, subclasses=None,
        shuffle=True, test=False
    ):
        """
        - results_dir: Path (either absolute or relative) to directory where
        output files should be saved. This directory will be created by the
        program and so should not already exist.
        - fluor_data: DataFrame of (scaled, but not standardised) fluorescence
        readings (output from parse_array_data class)
        - classes: List defining class labels associated with the samples, e.g.
        for analysis of tea types an example list would be ['Green', 'Green',
        'Black', 'Black', 'Grey', 'Grey']. If not defined, code will select the
        column labelled 'Analyte' in the fluor_data DataFrame. Default set to
        None (i.e. not defined).
        - subclasses: List defining subclass labels associated with the samples
        (if such subclass labels exist), e.g. for analysis of tea types an
        example list would be ['Green_Tetleys', 'Green_Pukka',
        'Black_Yorkshire', 'Black_Yorkshire', 'Grey_Asda', 'Grey_Pukka'].
        Default set to None (i.e. not defined).
        - shuffle: Boolean, setting to True directs the program to randomly
        shuffle the input data before ML (to ensure removal of any ordering
        effects in the input data.) Default set to True.
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False
        """

        DefData.__init__(self, results_dir, test)

        # Checks that if subclasses are defined, classes are also defined
        if not subclasses is None:
            if classes is None:
                raise TypeError(
                    'If "subclasses" is set to a value other than None, then '
                    '"classes" must also be set to a value other than None'
                )
        # Checks that classes argument is the correct type
        if not classes is None:
            if not type(classes) is list:
                raise TypeError(
                    'Expect "classes" argument to be set either to None or to a'
                    ' list'
                )
        # Checks that subclasses argument is the correct type
        if not subclasses is None:
            if not type(subclasses) is list:
                raise TypeError(
                    'Expect "subclasses" argument to be set either to None or '
                    'to a list'
                )
            for subclass in subclasses:
                if type(subclass) == str:
                    if subclass.count('_') != 1:
                        raise ValueError(
                            'Entries in subclass list should be formatted as '
                            '"class_subclass" (in which neither "class" nor '
                            '"subclass" contains the character "_")'
                        )
                    else:
                        if not subclass.split('_')[0] in classes:
                            raise ValueError(
                                'Entries in subclass list should be formatted '
                                'as "class_subclass" (in which neither "class" '
                                'nor "subclass" contains the character "_")'
                            )

        # Defines classes and subclasses lists if not specified by the user
        if classes is None:
            try:
                classes = fluor_data['Analyte'].tolist()
            except KeyError:
                raise KeyError(
                    'No "Analyte" column detected in input dataframe - if you '
                    'do not define the "classes" argument, the input dataframe'
                    ' must contain an "Analyte" column'
            )
        if subclasses is None:
            subclasses = [np.nan for n in range(fluor_data.shape[0])]
        if len(classes) != fluor_data.shape[0]:
            raise ValueError(
                'Mismatch between number of entries in the input dataframe and '
                'the "classes" list'
            )
        if len(subclasses) != fluor_data.shape[0]:
            raise ValueError(
                'Mismatch between number of entries in the input dataframe and '
                'the "subclasses" list'
            )

        # Ensures that no columns in the input dataframe will be overwritten
        # when 'Classes' and 'Subclasses' columns are subsequently appended
        if any(x in fluor_data.columns for x in ['Classes', 'Subclasses']):
            raise NameError(
                'Please rename any columns in input dataframe labelled either '
                '"Classes" or "Subclasses", as these columns are added to the '
                'dataframe by the code during data processing'
            )
        class_df = pd.DataFrame({'Classes': classes,
                                 'Subclasses': subclasses})
        fluor_data = fluor_data.reset_index(drop=True)
        fluor_data = pd.concat(
            [fluor_data, class_df], axis=1
        ).reset_index(drop=True)

        # Checks that "shuffle" is a Boolean value
        if type(shuffle) != bool:
            raise TypeError(
                'Expect "shuffle" to be a Boolean value (True or False)'
            )
        # Randomly shuffles input data
        if shuffle is True:
            indices = [i for i in range(fluor_data.shape[0])]
            random.shuffle(indices)
            fluor_data = fluor_data.iloc[indices].reset_index(drop=True)

        # Defines object attributes used in ML functions
        self.classes = fluor_data['Classes'].to_numpy(dtype=str)
        self.sub_classes = fluor_data['Subclasses'].to_numpy(dtype=str)
        fluor_data = fluor_data.drop(['Classes', 'Subclasses'], axis=1)
        try:
            fluor_data = fluor_data.drop(['Analyte'], axis=1)
        except KeyError:
            pass
        self.fluor_data = fluor_data
        self.features = self.fluor_data.columns.tolist()
        self.x = self.fluor_data.to_numpy()
        self.y = copy.deepcopy(self.classes)

        # Checks there are no NaN values in the input data
        if self.fluor_data.astype('object').isna().any(axis=None):
            raise ValueError(
                'NaN detected in input dataframe'
            )
        if np.dtype('O') in self.fluor_data.dtypes.tolist():
            raise ValueError(
                'Non-numeric value detected in input dataframe'
            )
        if 'nan' in self.classes:
            raise ValueError(
                'NaN detected in class values'
            )

        if 'nan' in self.sub_classes:
            nan_array = np.array(['nan' for n in range(self.fluor_data.shape[0])])
            if (self.sub_classes == nan_array).all(axis=None):
                self.sub_classes = None
            else:
                raise ValueError(
                    'NaN detected in subclass values'
                )

    def split_train_test_data_random(
        self, x, y, const_split=False, percent_test=0.2, test=False
    ):
        """
        Splits data at random into training and test sets

        Input
        ----------
        - x: Numpy array of x values
        - y: Numpy array of y values
        - const_split: Boolean, if set to True and splits is set to None,
        function will generate splits using a fixed random_state (=>
        reproducible results when the function is re-run)
        - percent_test: Float in the range 0-0.5, defines the fraction of the
        total data to be set aside for model testing
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - split: [train_indices, test_indices], where train_indices is an array
        of the rows in x and y that form the training dataset, and test_indices
        is an array of the rows in x and y that form the testing dataset
        """

        if type(x) != np.ndarray:
            raise TypeError('Expect "x" to be an array of x values')
        if type(y) != np.ndarray:
            raise TypeError('Expect "y" to be an array of y values')
        if pd.DataFrame(x, dtype=object).isna().any(axis=None):
            raise ValueError('NaN value(s) detected in "x" data')
        if pd.DataFrame(y, dtype=object).isna().any(axis=None):
            raise ValueError('NaN value(s) detected in "y" data')
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                'Mismatch in the dimensions of the input "x" and "y" values'
            )
        if type(const_split) != bool:
            raise TypeError(
                'Expect "const_split" to be a Boolean (True or False)'
            )
        if not type(percent_test) in [int, float]:
            raise TypeError(
                '"percent_test" argument should be set to a float in the range '
                '0 - 0.5'
            )
        else:
            if percent_test < 0 or percent_test > 0.5:
                raise ValueError(
                    '"percent_test" argument should be set to a float in the '
                    'range 0 - 0.5'
                )

        if percent_test > 0:
            if test is True or const_split is True:
                skf = StratifiedKFold(
                    n_splits=round(1/percent_test), shuffle=True, random_state=0
                )
            else:
                skf = StratifiedKFold(
                    n_splits=round(1/percent_test), shuffle=True
                )
            split = [sub_split for sub_split in list(skf.split(X=x, y=y))[0]]
        elif percent_test == 0:
            split = [np.array([n for n in range(x.shape[0])]), np.array([])]

        return split

    def split_train_test_data_user_defined(self, subclasses, test_subclasses):
        """
        Splits data into training and test sets defined by the user. Will only
        run if self.sub_classes is not set to None.

        Input
        ----------
        - subclasses: Numpy array of subclass values
        - test_subclasses: Numpy array of the subclasses to be separated out
        for the test set. Must not include all of the subclasses in a class.

        Output
        ----------
        - split: [train_indices, test_indices], where train_indices is an array
        of the rows in x and y that form the training dataset, and test_indices
        is an array of the rows in x and y that form the testing dataset
        """

        if type(subclasses) != np.ndarray:
            raise TypeError(
                'Expect "subclasses" to be a (1D) array of subclass values'
            )
        if len(subclasses.shape) != 1:
            raise ValueError('Expect "subclasses" to be a 1D array')
        if pd.DataFrame(subclasses, dtype=object).isna().any(axis=None):
            raise ValueError('NaN value(s) detected in "subclasses" array')

        if type(test_subclasses) != np.ndarray:
            raise TypeError(
                'Expect "test_subclasses" argument to be a (1D) array of the '
                'subclass values that should be separated out into the test set'
            )
        if len(test_subclasses.shape) != 1:
            raise ValueError('Expect "test_subclasses" to be a 1D array')
        if pd.DataFrame(test_subclasses, dtype=object).isna().any(axis=None):
            raise ValueError('NaN value(s) detected in "test_subclasses" array')

        for subclass in test_subclasses:
            if not subclass in subclasses:
                raise ValueError(
                    'Not all of the entries in the "test_subclasses" array are '
                    'found in the "subclasses" array. Expect "test_subclasses" '
                    'argument to be a (1D) array of the subclass values that '
                    'should be separated out into the test set'
                )

        train_indices = []
        test_indices = []

        for n in range(subclasses.shape[0]):
            if subclasses[n] in test_subclasses:
                test_indices.append(n)
            else:
                train_indices.append(n)

        split = [np.array(train_indices), np.array(test_indices)]

        return split

    def calc_feature_correlations(
        self, fluor_data=None, correlation_coeff='kendall', plt_name='',
        abs_vals=True, test=False
    ):
        """
        Calculates correlation coefficient values between all 2-way combinations
        of features, and plots a heatmap.

        Input
        ----------
        - fluor_data: DataFrame of fluorescence readings. By default set to
        self.fluor_data
        - correlation_coeff: String defining the coefficient coefficient to be
        calculate between the data, by default is set to Kendall's tau. Can
        alternatively be set to "pearson" (not recommended) or "spearman".
        - plt_name: Prefix to append to name of the saved plot
        - abs_vals: Boolean, if set to True colours the the heatmap by absolute
        correlation coefficient values, if set to False colours by the raw
        values. (Colouring by the absolute values prevents strong negative
        correlations from appearing to be weakly correlated when quickly
        glancing at the heatmap.)
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - correlation matrix: DataFrame of Kendall tau rank correlation
        coefficient values for all pairwise combinations of features
        """

        # Checks arguments are suitable for running the function
        if fluor_data is None:
            fluor_data = copy.deepcopy(self.fluor_data)

        if type(fluor_data) != pd.core.frame.DataFrame:
            raise TypeError(
                '"fluor_data" should be a dataframe of fluorescence readings'
            )

        if fluor_data.applymap(
            lambda x: isinstance(x, (int, float))).all(axis=None, skipna=False
        ) is np.bool_(False):
            raise ValueError(
                'Non-numeric value(s) in "fluor_data" - expect all values in '
                '"fluor_data" to be integers / floats'
            )

        if fluor_data.isna().any(axis=None):
            raise ValueError(
                'NaN value(s) found in "fluor_data"'
            )

        if not correlation_coeff in ['kendall', 'spearman', 'pearson']:
            raise ValueError(
                'Value specified for "correlation_coeff" not recognised - '
                'should be set to "kendall", "spearman" or "pearson"'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        if type(abs_vals) != bool:
            raise TypeError(
                '"abs_vals" should be a Boolean value'
            )

        # Calculates the correlation coefficient
        correlation_matrix = copy.deepcopy(fluor_data).corr(
            method=correlation_coeff
        )
        if abs_vals is True:
            correlation_matrix = correlation_matrix.abs()

        plt.clf()
        if abs_vals is False:
            heatmap = sns.heatmap(
                data=correlation_matrix, cmap='RdBu_r', annot=True,
                xticklabels=True, yticklabels=True, fmt='.3f'
            )
        else:
            heatmap = sns.heatmap(
                data=correlation_matrix, cmap='RdBu_r', annot=True,
                xticklabels=True, yticklabels=True, fmt='.3f', vmin=0, vmax=1
            )
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        plt.savefig('{}/{}Feature_correlations_Kendall_tau_rank.svg'.format(
            self.results_dir, plt_name
        ))
        if test is False:
            plt.show()

        return correlation_matrix

    def calc_feature_importances_kbest(
        self, x=None, y=None, features=None, method_classif='f_classif',
        num_repeats=1000, scale=True, plt_name='', test=False
    ):
        """
        Runs a univariate statistical test (either f_classif (ANOVA F-statistic,
        parametric) or mutual_info_classif (mutual information, non-parametric))
        between x and y to calculate the importances of the different input
        features, and plots these values. Test is run multiple (default 1000)
        times on bootstrapped data, results are then displayed as the median and
        95% confidence limits calculated from the 2.5th and 97.5th percentiles.
        N.B. Assumes that the features are independent of one another.

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of y values. By default set to self.y.
        - features: List of input features corresponding to the columns in x. By
        default set to self.features.
        - method_classif: Either 'f_classif' or 'mutual_info_classif'. From the
        scikit-learn docs - 'The methods based on F-test (f_classif) estimate
        the degree of linear dependency between two random variables. On the
        other hand, mutual information methods can capture any kind of
        statistical dependency, but being nonparametric, they require more
        samples for accurate estimation.' By default set to f_classif
        - num_repeats: The number of times to repeat the test. By default set
        to 1000.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the
        importance scores.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - importance_df: DataFrame of features and their mean average importance
        scores
        - univ_feature_importances: Dictionary of features and their importance
        scores across num_repeats
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if features is None:
            features = copy.deepcopy(self.features)

        if type(x) != np.ndarray:
            raise TypeError(
                'Expect "x" to be a (2D) array of fluorescence readings'
            )

        if len(x.shape) != 2:
            raise ValueError(
                'Expect "x" to be a (2D) array of fluorescence readings'
            )

        if type(y) != np.ndarray:
            raise TypeError(
                'Expect "y" to be a (1D) array of class labels'
            )

        if len(y.shape) != 1:
            raise ValueError(
                'Expect "y" to be a (1D) array of class labels'
            )

        if type(features) != list:
            raise TypeError(
                'Expect "features" to be a list of the column ids in "x"'
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                'Mismatch between the number of rows in "x" and the number of '
                'entries in "y"'
            )

        if x.shape[1] != len(features):
            raise ValueError(
                'Mismatch between the number of columns in "x" and the number '
                'of column ids in "features"'
            )

        if not method_classif in [
            f_classif, mutual_info_classif, 'f_classif', 'mutual_info_classif'
        ]:
            raise ValueError(
                '"method_classif" should be set to either "f_classif" or '
                '"mutual_info_classif"'
            )

        if method_classif == 'f_classif':
            method_classif = f_classif
        elif method_classif == 'mutual_info_classif':
            method_classif = mutual_info_classif

        if type(num_repeats) != int:
            raise TypeError(
                '"num_repeats" should be set to a positive integer value'
            )
        else:
            if num_repeats <= 0:
                raise ValueError(
                    '"num_repeats" should be set to a positive integer value'
                )

        if type(scale) != bool:
            raise TypeError(
                '"scale" should be set to a Boolean value'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        # Runs SelectKBest
        univ_feature_importances = OrderedDict()
        for col in features:
            univ_feature_importances[col] = [np.nan for n in range(num_repeats)]

        for n in range(num_repeats):
            # Uses bootstrapping to create a "new" dataset
            temp_x, temp_y = bootstrap_data(x, y, features, scale, test)

            model = SelectKBest(score_func=method_classif, k='all')
            model.fit(X=temp_x, y=temp_y)
            total_y = np.sum(model.scores_)
            norm_y = model.scores_ / total_y

            for col, importance in enumerate(norm_y):
                col = features[col]
                univ_feature_importances[col][n] = importance

        plt_name = '{}_KBest'.format(plt_name)
        if test is False:
            importance_df = make_feat_importance_plots(
                univ_feature_importances, self.results_dir, plt_name, test
            )
        else:
            (
                importance_df, cols, cols_all, all_vals, median_vals,
                lower_conf_limit_vals, upper_conf_limit_vals
            ) = make_feat_importance_plots(
                    univ_feature_importances, self.results_dir, plt_name, test
                )

        return importance_df, univ_feature_importances

    def calc_feature_importances_tree(
        self, x=None, y=None, features=None, num_repeats=1000, scale=True,
        plt_name='', test=False
    ):
        """
        Fits an ExtraTrees classifier to the data (a number of randomised
        decision trees are fitted to various different sub-samples of the data,
        these "sub-trees" are then averaged). Feature importance scores can be
        calculated from the average contribution which each feature makes to the
        predictive accuracy of the sub-trees in which it is included. Classifier
        is fitted multiple (default 1000) times on bootstrapped data, results
        are then displayed as the median and 95% confidence limits calculated
        from the 2.5th and 97.5th percentiles.
        N.B. Does not assume that the features are independent of one another,
        but equally does not fully take into account correlations between
        features (the user should run permutation analysis instead if feature
        correlation is a real concern).

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of y values. By default set to self.y.
        - features: List of input features corresponding to the columns in x. By
        default set to self.features.
        - num_repeats: The number of times to repeat the test. By default set
        to 1000.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the
        importance scores.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - importance_df: DataFrame of features and their mean average importance
        scores
        - tree_feature_importances: Dictionary of features and their importance
        scores across num_repeats
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if features is None:
            features = copy.deepcopy(self.features)

        if type(x) != np.ndarray:
            raise TypeError(
                'Expect "x" to be a (2D) array of fluorescence readings'
            )

        if len(x.shape) != 2:
            raise ValueError(
                'Expect "x" to be a (2D) array of fluorescence readings'
            )

        if type(y) != np.ndarray:
            raise TypeError(
                'Expect "y" to be a (1D) array of class labels'
            )

        if len(y.shape) != 1:
            raise ValueError(
                'Expect "y" to be a (1D) array of class labels'
            )

        if type(features) != list:
            raise TypeError(
                'Expect "features" to be a list of the column ids in "x"'
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                'Mismatch between the number of rows in "x" and the number of '
                'entries in "y"'
            )

        if x.shape[1] != len(features):
            raise ValueError(
                'Mismatch between the number of columns in "x" and the number '
                'of column ids in "features"'
            )

        if type(num_repeats) != int:
            raise TypeError(
                '"num_repeats" should be set to a positive integer value'
            )
        else:
            if num_repeats <= 0:
                raise ValueError(
                    '"num_repeats" should be set to a positive integer value'
                )

        if type(scale) != bool:
            raise TypeError(
                '"scale" should be set to a Boolean value'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        # Fits ExtraTrees classifiers
        tree_feature_importances = OrderedDict()
        for col in features:
            tree_feature_importances[col] = [np.nan for n in range(num_repeats)]

        for n in range(num_repeats):
            # Uses bootstrapping to create a "new" dataset
            temp_x, temp_y = bootstrap_data(x, y, features, scale, test)

            if test is False:
                model = ExtraTreesClassifier()
            else:
                model = ExtraTreesClassifier(random_state=1)
            model.fit(X=temp_x, y=temp_y)
            feature_importances = model.feature_importances_

            for col, importance in enumerate(model.feature_importances_):
                col = features[col]
                tree_feature_importances[col][n] = importance

        plt_name = '{}_Tree'.format(plt_name)
        if test is False:
            importance_df = make_feat_importance_plots(
                tree_feature_importances, self.results_dir, plt_name, test
            )
        else:
            (
                importance_df, cols, cols_all, all_vals, median_vals,
                lower_conf_limit_vals, upper_conf_limit_vals
            ) = make_feat_importance_plots(
                    tree_feature_importances, self.results_dir, plt_name, test
                )

        return importance_df, tree_feature_importances

    def calc_feature_importances_permutation(
        self, x=None, y=None, features=None, classifier=AdaBoostClassifier,
        parameters={'n_estimators': [10, 30, 100, 300, 1000]},
        model_metric='accuracy', num_repeats=1000, scale=True, plt_name='',
        test=False
    ):
        """
        Runs permutation analysis with a classifier to calculate feature
        importances (for each feature in turn, the performance of a classifier
        trained with and without that feature on the input dataset is compared -
        the importance score is the difference in performance between the two
        models). Analysis is run multiple (default 1000) times on bootstrapped
        data, results are then displayed as the median and 95% confidence limits
        calculated from the 2.5th and 97.5th percentiles.
        N.B. Does not assume that the features are independent of one another,
        but equally does not fully take into account correlations between
        features (the user should run permutation analysis instead if feature
        correlation is a real concern).

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of y values. By default set to self.y.
        - features: List of input features corresponding to the columns in x. By
        default set to self.features.
        - classifier: The type of classifier to train. By default set to
        AdaBoostClassifier. BEWARE of selecting a classifier that is slow to
        train, especially if num_repeats is set to a high value.
        - parameters: Dictionary of parameter values to optimise (via grid
        search) for the classifier. Also include in this dictionary any fixed,
        non-default parameter values that you want to use. By default set to a
        sensible range of n_estimators to try for an AdaBoostClassifier. BEWARE
        of setting to an extensive range of parameters to test, as the function
        will run *very* slowly.
        - model_metric: The metric to use to measure classifier performance. By
        default set to accuracy.
        - num_repeats: The number of times to repeat the test. By default set
        to 1000.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the
        importance scores.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - importance_df: DataFrame of features and their mean average importance
        scores
        - permutation_feature_importances: Dictionary of features and their
        importance scores across num_repeats
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if features is None:
            features = copy.deepcopy(self.features)

        if type(x) != np.ndarray:
            raise TypeError(
                'Expect "x" to be a (2D) array of fluorescence readings'
            )

        if len(x.shape) != 2:
            raise ValueError(
                'Expect "x" to be a (2D) array of fluorescence readings'
            )

        if type(y) != np.ndarray:
            raise TypeError(
                'Expect "y" to be a (1D) array of class labels'
            )

        if len(y.shape) != 1:
            raise ValueError(
                'Expect "y" to be a (1D) array of class labels'
            )

        if type(features) != list:
            raise TypeError(
                'Expect "features" to be a list of the column ids in "x"'
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                'Mismatch between the number of rows in "x" and the number of '
                'entries in "y"'
            )

        if x.shape[1] != len(features):
            raise ValueError(
                'Mismatch between the number of columns in "x" and the number '
                'of column ids in "features"'
            )

        if not type(parameters) in [dict, OrderedDict]:
            raise TypeError(
                'Expect "parameters" to be a dictionary of parameter names '
                '(keys) and arrays of values to consider for them (values) in a'
                ' grid search'
            )

        metrics_list = [
            'accuracy', 'balanced_accuracy', 'top_k_accuracy',
            'average_precision','neg_brier_score', 'f1', 'f1_micro', 'f1_macro',
            'f1_weighted','f1_samples', 'neg_log_loss', 'precision',
            'precision_micro','precision_macro', 'precision_weighted',
            'precision_samples', 'recall','recall_micro', 'recall_macro',
            'recall_weighted', 'recall_samples','jaccard', 'jaccard_micro',
            'jaccard_macro', 'jaccard_weighted','jaccard_samples', 'roc_auc',
            'roc_auc_ovr', 'roc_auc_ovo','roc_auc_ovr_weighted',
            'roc_auc_ovo_weighted'
        ]
        if type(model_metric) == sklearn.metrics._scorer._PredictScorer:
            pass
        else:
            if not model_metric in metrics_list:
                raise ValueError(
                    'Value provided for "model_metric" not recognised - please '
                    'specify one of the strings in the list below:\n'
                    '{}'.format(metrics_list)
                )

        if type(num_repeats) != int:
            raise TypeError(
                '"num_repeats" should be set to a positive integer value'
            )
        else:
            if num_repeats <= 0:
                raise ValueError(
                    '"num_repeats" should be set to a positive integer value'
                )

        if type(scale) != bool:
            raise TypeError(
                '"scale" should be set to a Boolean value'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        # Fits classifiers
        permutation_feature_importances = OrderedDict()
        for col in features:
            permutation_feature_importances[col] = [np.nan for n in range(num_repeats)]

        # For speed reasons, perform one grid search to obtain "optimal"
        # parameters on the original data, rather than re-running for each
        # bootstrapped dataset => greatly increases function speed whilst having
        # little effect upon performance (an OK set of parameter values is
        # expected to work well for all of the bootstrapped datasets)
        if test is False:
            orig_model = copy.deepcopy(classifier)()
        else:
            try:
                orig_model = copy.deepcopy(classifier)(random_state=1)
            except TypeError:
                orig_model = copy.deepcopy(classifier)()
        orig_grid_search = GridSearchCV(
            estimator=orig_model, param_grid=parameters, error_score=np.nan,
            scoring=model_metric
        )

        if scale is True:
            scaled_x = RobustScaler().fit_transform(x)
            orig_grid_search.fit(X=scaled_x, y=y)
        else:
            orig_grid_search.fit(X=x, y=y)
        best_params = orig_grid_search.best_params_

        for n in range(num_repeats):
            # Uses bootstrapping to create a "new" dataset
            temp_x, temp_y = bootstrap_data(x, y, features, scale, test)
            temp_x = temp_x.to_numpy()
            temp_y = np.array(temp_y)

            if test is True:
                best_params['random_state'] = 1
            model = copy.deepcopy(classifier)(**best_params)
            model.fit(temp_x, temp_y)

            if test is False:
                results = permutation_importance(
                    model, temp_x, temp_y, scoring=model_metric, n_jobs=-1
                )
            else:
                results = permutation_importance(
                    model, temp_x, temp_y, scoring=model_metric, n_jobs=-1,
                    random_state=1
                )

            for col, importance in enumerate(results.importances_mean):
                col = features[col]
                permutation_feature_importances[col][n] = importance

        plt_name = '{}_Permutation'.format(plt_name)
        if test is False:
            importance_df = make_feat_importance_plots(
                permutation_feature_importances, self.results_dir, plt_name,
                test
            )
        else:
            (
                importance_df, cols, cols_all, all_vals, median_vals,
                lower_conf_limit_vals, upper_conf_limit_vals
            ) = make_feat_importance_plots(
                    permutation_feature_importances, self.results_dir, plt_name,
                    test
                )

        return importance_df, permutation_feature_importances

    def run_pca(self, fluor_data=None, scale=True, plt_name='', test=False):
        """
        Runs Principal Component Analysis and makes scatter plot of number of
        components vs. amount of information captured, calculates the
        contribution of each feature to each principal component.

        Input
        ----------
        - fluor_data: DataFrame of fluorescence readings. By default set to
        self.fluor_data
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the PCA.
        - plt_name: Prefix to append to the names of the saved plot
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - model: PCA model fitted to x values
        """

        # Checks argument values are suitable for running the function
        if fluor_data is None:
            fluor_data = copy.deepcopy(self.fluor_data)

        if type(fluor_data) != pd.DataFrame:
            raise TypeError(
                'Expect "fluor_data" to be dataframe of fluorescence readings'
            )

        if fluor_data.applymap(
            lambda x: isinstance(x, (int, float))).all(axis=None, skipna=False
        ) is np.bool_(False):
            raise ValueError(
                'Non-numeric value(s) in "fluor_data" - expect all values in '
                '"fluor_data" to be integers / floats'
            )

        if type(scale) != bool:
            raise TypeError('"scale" should be set to a Boolean value')

        if type(plt_name) != str:
            raise TypeError('"plt_name" should be a string value')

        # Runs PCA
        x = fluor_data.to_numpy()
        if scale is True:
            x = RobustScaler().fit_transform(x)

        model = PCA()
        model.fit(x)
        max_num_components = len(model.explained_variance_ratio_)

        plt.clf()
        x_labels = np.linspace(1, max_num_components, max_num_components)
        lineplot = sns.lineplot(
            x=x_labels, y=np.cumsum(model.explained_variance_ratio_), marker='o'
        )
        plt.xticks(
            np.arange(1, x_labels.shape[0]+1), x_labels, rotation='vertical'
        )
        plt.yticks(rotation='horizontal')
        plt.savefig('{}/{}PCA_scree_plot.svg'.format(self.results_dir, plt_name))
        if test is False:
            plt.show()

        # Calculates contribution of each feature to each principal component
        features = list(fluor_data.columns)
        pca_components = OrderedDict(
            {'Component': [int(n) for n in range(1, len(features)+1)]}
        )
        for index, feature in enumerate(features):
            pca_components[feature] = model.components_[:,index]
        pca_components = pd.DataFrame(pca_components)
        pca_components = pca_components.set_index('Component', drop=True)

        sns.heatmap(
            data=pca_components.abs(), cmap='RdBu_r', annot=True,
            xticklabels=True, yticklabels=True
        )
        plt.savefig('{}/{}PCA_component_heatmap.svg'.format(
            self.results_dir, plt_name
        ))
        if test is False:
            plt.show()

        return model, pca_components

    def plot_scatter_on_pca_axes(
        self, x=None, y=None, subclasses=None, num_dimensions=2, scale=True,
        plt_name='', test=False
    ):
        """
        Generates scatter plot of input data on axes corresponding to the first
        two (2D plot)/three (3D) principal components calculated for the dataset

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of class (y) values. By default set to self.y.
        - subclasses: Numpy array of subclasses. By default set to
        self.sub_classes.
        - num_dimensions: The number of PCA dimensions to incorporate in the
        scatter plot. Must be set to either 2 or 3, by default is set to 2.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the PCA.
        - plt_name: Prefix to append to the names of the saved plot
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if subclasses is None:
            subclasses = copy.deepcopy(self.sub_classes)

        if type(x) != np.ndarray:
            raise TypeError(
                'Expect "x" to be a (2D) array of fluorescence readings'
            )

        if len(x.shape) != 2:
            raise ValueError(
                'Expect "x" to be a (2D) array of fluorescence readings'
            )

        if type(y) != np.ndarray:
            raise TypeError(
                'Expect "y" to be a (1D) array of class labels'
            )

        if len(y.shape) != 1:
            raise ValueError(
                'Expect "y" to be a (1D) array of class labels'
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                'Mismatch between the number of rows in "x" and the number of '
                'entries in "y"'
            )

        if not subclasses is None:
            if type(subclasses) != np.ndarray:
                raise TypeError(
                    'Expect "subclasses" to be set either to None, or to a (1D)'
                    ' array of the subclasses present in the dataset'
                )

            if len(subclasses.shape) != 1:
                raise ValueError(
                    'Expect "subclasses" to be set either to None, or to a (1D)'
                    ' array of the subclasses present in the dataset'
                )

            if x.shape[0] != subclasses.shape[0]:
                raise ValueError(
                    'Mismatch between the number of rows in "x" and the number '
                    'of entries in "subclasses"'
                )

        if not num_dimensions in [2, 3]:
            raise ValueError('Expect "num_dimensions" to be set to 2 or 3')

        if type(scale) != bool:
            raise TypeError('"scale" should be a Boolean (True or False)')

        if type(plt_name) != str:
            raise TypeError('"plt_name" should be a string value')

        # Selects marker colour and symbol
        colours = [key for key, val in BASE_COLORS.items()]
        extra_colours = [key for key, val in CSS4_COLORS.items()]
        colours += extra_colours[::-1]
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'D', 'd', 'P', 'X',
                   'h', 'H']*10

        categories = []
        if subclasses is None:
            categories = ['{}_'.format(cat) for cat in sorted(list(set(list(y))))]
        else:
            categories = sorted(list(set(list(subclasses))))

        cat_class_colours = {}
        colour_count = 0
        for cat in categories:
            if cat.count('_') != 1:
                raise ValueError(
                    'Character "_" found in subclass {} more/less than '
                    'once'.format(cat)
                )
            cat_class = cat.split('_')[0]
            if not cat_class in cat_class_colours:
                cat_class_colours[cat_class] = colours[colour_count]
                colour_count += 1

        cat_markers = {}
        for cat_1_class in cat_class_colours.keys():
            marker_count = 0
            for cat_2 in categories:
                cat_2_class = cat_2.split('_')[0]
                if cat_1_class == cat_2_class:
                    cat_markers[cat_2.rstrip('_')] = markers[marker_count]
                    marker_count += 1

        cat_colours = {}
        for cat in cat_markers.keys():
            cat_class = cat.split('_')[0]
            colour = cat_class_colours[cat_class]
            cat_colours[cat.rstrip('_')] = colour

        # Runs PCA
        if scale is True:
            x = RobustScaler().fit_transform(x)
        pca = PCA(n_components=num_dimensions)
        X_reduced = pca.fit_transform(X=x)

        # Generates scatter plot
        fig = plt.figure()
        if subclasses is None:
            plot_y = copy.deepcopy(y)
        else:
            plot_y = copy.deepcopy(subclasses)

        if num_dimensions == 2:
            ax = fig.add_subplot(111)
            for i, y_val in np.ndenumerate(plot_y):
                scatter = ax.scatter(
                    X_reduced[i[0],0], X_reduced[i[0],1], c=cat_colours[y_val],
                    marker=cat_markers[y_val]
                )
            ax.set_xlabel('Principal component 1')
            ax.set_ylabel('Principal component 2')
        elif num_dimensions == 3:
            ax = fig.add_subplot(111, projection='3d')
            for i, y_val in np.ndenumerate(plot_y):
                scatter = ax.scatter(
                    X_reduced[i[0],0], X_reduced[i[0],1], X_reduced[i[0],2],
                    c=cat_colours[y_val], marker=cat_markers[y_val]
                )
            ax.set_xlabel('Principal component 1')
            ax.set_ylabel('Principal component 2')
            ax.set_zlabel('Principal component 3')

        legend_elements = []
        for cat, colour in cat_colours.items():
            marker = cat_markers[cat]
            legend_elements.append(
                Line2D([0], [0], marker=marker, color=colour, label=cat,
                markerfacecolor=colour)
            )
        ax.legend(
            handles=legend_elements, bbox_to_anchor=(1.05, 1), title='Classes'
        )

        plt.savefig('{}/{}_{}_PCA_plot.svg'.format(
            self.results_dir, plt_name, str(num_dimensions)
        ))
        if test is False:
            plt.show()
        else:
            return (cat_class_colours, cat_markers, cat_colours, X_reduced)

    def define_fixed_model_params(self, classifier):
        """
        For the 8 default ML algorithms run by this code (LogisticRegression,
        KNeighborsClassifier, GaussianNB, LinearDiscriminantAnalysis, LinearSVC,
        SVC, AdaBoostClassifier and DummyClassifier), defines a dictionary of
        hyperparameters and their corresponding values that will remain fixed
        throughout optimisation and training

        Input
        ----------
        - classifier: The selected ML algorithm,
        e.g. sklearn.ensemble.AdaBoostClassifier()

        Output
        ----------
        - params: Dictionary of fixed hyperparameters
        """

        if type(classifier).__name__ == 'LogisticRegression':
            params = OrderedDict({'n_jobs': -1,
                                  'max_iter': 1000})
        elif type(classifier).__name__ == 'KNeighborsClassifier':
            params = OrderedDict({'metric': 'minkowski',
                                  'n_jobs': -1})
        elif type(classifier).__name__ == 'LinearSVC':
            params = OrderedDict({'dual': False,
                                  'max_iter': 10000})  # Change back to True
            # (= default) if n_samples < n_features
        elif type(classifier).__name__ == 'SVC':
            params = OrderedDict()
        elif type(classifier).__name__ == 'AdaBoostClassifier':
            params = OrderedDict()
        elif type(classifier).__name__ == 'GaussianNB':
            params = OrderedDict()
        elif type(classifier).__name__ == 'LinearDiscriminantAnalysis':
            params = OrderedDict()
        elif type(classifier).__name__ == 'DummyClassifier':
            params = OrderedDict({'strategy': 'prior'})
        else:
            raise TypeError(
                'Unrecognised value provided for "classifier". Expect '
                '"classifier" to be one of:\n'
                'sklearn.linear_model.LogisticRegression()\n'
                'sklearn.neighbors.KNeighborsClassifier()\n'
                'sklearn.svm.LinearSVC()\n'
                'sklearn.svm.SVC()\n'
                'sklearn.ensemble.AdaBoostClassifier()\n'
                'sklearn.naive_bayes.GaussianNB()\n'
                'sklearn.discriminant_analysis.LinearDiscriminantAnalysis()\n'
                'sklearn.dummy.DummyClassifier()'
            )

        return params

    def define_tuned_model_params(self, classifier, x_train, n_folds=5):
        """
        For the 8 default ML algorithms run by this code (LogisticRegression,
        KNeighborsClassifier, GaussianNB, LinearDiscriminantAnalysis, LinearSVC,
        SVC, AdaBoostClassifier and DummyClassifier), returns dictionary of a
        sensible range of values for variable hyperparameters to be tested in
        randomised / grid search

        Input
        ----------
        - classifier: The selected ML algorithm,
        e.g. sklearn.ensemble.AdaBoostClassifier()
        - x_train: Numpy array of x values of training data
        - n_folds: The number of folds to be used in (stratified) k-folds
        cross-validation. By default set to 5

        Output
        ----------
        - params: Dictionary of parameter ranges that can be fed into
        RandomizedSearchCV or GridSearchCV
        """

        if type(x_train) != np.ndarray:
            raise TypeError(
                '"x_train" should be a (2D) array of fluoresence readings'
            )

        if type(n_folds) != int:
            raise TypeError(
                '"n_folds" should be set to a positive integer value'
            )
        else:
            if n_folds < 1:
                raise ValueError(
                    '"n_folds" should be set to a positive integer value'
                )

        shape = x_train.shape[0]
        if type(classifier).__name__ == 'LogisticRegression':
            params = OrderedDict({
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'sag', 'saga', 'newton-cg', 'lbfgs'],
                'multi_class': ['ovr', 'multinomial'],
                'C': np.logspace(-3, 5, 17)
            })
        elif type(classifier).__name__ == 'KNeighborsClassifier':
            if (1/n_folds)*shape < 2:
                raise AlgorithmError(
                    'Too few data points in dataset to run k nearest neighbours'
                )
            else:
                neighbours = np.array(range(2, int((1/n_folds)*shape), 1))
                params = OrderedDict({
                    'n_neighbors': neighbours,
                    'weights': ['uniform', 'distance'],
                    'p': np.array([1, 2])
                })
        elif type(classifier).__name__ == 'LinearSVC':
            params = OrderedDict({'C': np.logspace(-5, 15, num=21, base=2)})
        elif type(classifier).__name__ == 'SVC':
            # For speed reasons (some kernels take a prohibitively long time to
            # train) am sticking with the default kernel ('rbf')
            params = OrderedDict({
                'C': np.logspace(-5, 15, num=21, base=2),
                'gamma': np.logspace(-15, 3, num=19, base=2),
                'kernel': ['rbf']
            })
        elif type(classifier).__name__ == 'AdaBoostClassifier':
            if (1/n_folds)*shape < 2:
                raise AlgorithmError(
                    'Too few data points in dataset to use AdaBoost classifier'
                )
            else:
                n_estimators = np.array([int(x) for x in np.logspace(1, 4, 7)])
                params = OrderedDict({'n_estimators': n_estimators})
        elif type(classifier).__name__ == 'GaussianNB':
            params = OrderedDict()
        elif type(classifier).__name__ == 'LinearDiscriminantAnalysis':
            params = OrderedDict()
        elif type(classifier).__name__ == 'DummyClassifier':
            params = OrderedDict()
        else:
            raise TypeError(
                'Unrecognised value provided for "classifier". Expect '
                '"classifier" to be one of:\n'
                'sklearn.linear_model.LogisticRegression()\n'
                'sklearn.neighbors.KNeighborsClassifier()\n'
                'sklearn.svm.LinearSVC()\n'
                'sklearn.svm.SVC()\n'
                'sklearn.ensemble.AdaBoostClassifier()\n'
                'sklearn.naive_bayes.GaussianNB()\n'
                'sklearn.discriminant_analysis.LinearDiscriminantAnalysis()\n'
                'sklearn.dummy.DummyClassifier()'
            )

        return params

    def flag_extreme_params(self, best_params, poss_params, test=False):
        """
        Flags a warning if hyperparameter optimisation selects a value at the
        extreme end of an input (numerical) range

        Input
        ----------
        - best_params: Dictionary of 'optimal' parameters returned by
        hyperparameter optimisation algorithm (e.g. RandomizedSearchCV or
        GridSearchCV)
        - poss_params: Dictionary of hyperparameter ranges fed into the
        hyperparameter optimisation algorithm, such as that returned by
        define_model_params
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False
        """

        if not type(best_params) in [dict, OrderedDict]:
            raise TypeError(
                'Expect "best_params" to be a dictionary of "optimal" parameter'
                ' values returned after running an algorithm such as '
                'RandomizedSearchCV or GridSearchCV'
            )
        if not type(poss_params) in [dict, OrderedDict]:
            raise TypeError(
                'Expect "poss_params" to be the dictionary of parameter ranges '
                'fed into the optimisation algorithm, such as that returned by '
                'define_model_params function'
            )
        if best_params.keys() != poss_params.keys():
            raise ValueError(
                'Mismatch in the keys in "best_params" and "poss_params"'
            )

        for param, best_val in best_params.items():
            poss_vals = poss_params[param]
            # If clauses are at separate indentations to avoid raising errors
            if isinstance(poss_vals, (list, np.ndarray)):
                if (
                        all(isinstance(poss_val, (int, float)) for poss_val in poss_vals)
                    and len(poss_vals) > 2
                ):
                    if best_val in [poss_vals[0], poss_vals[-1]]:
                        warning = (
                            '\x1b[31m WARNING: Optimal value selected for {} is'
                            ' at the extreme of the range tested \033[0m '.format(param)
                        )
                        warning += '\nRange tested: {}\nValue selected: {}\n\n'.format(
                            list(poss_vals), best_val
                        )
                        if test is False:
                            print(warning)
                        else:
                            return warning

    def conv_resampling_method(self, resampling_method, const_split, test=False):
        """
        Converts resampling method name (a string) into an object

        Input
        ----------
        - resampling_method: Name of resampling method (string)
        - const_split: Boolean, if set to True and splits is set to None,
        function will generate splits using a fixed random_state (=>
        reproducible results when the function is re-run)
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        resampling_obj: Resampling method (object)
        """

        if test is True or const_split is True:
            if resampling_method == 'no_balancing':
                resampling_obj = None
            elif resampling_method == 'max_sampling':
                resampling_obj = RandomOverSampler(
                    sampling_strategy='not majority', random_state=1
                )
            elif resampling_method == 'smote':
                resampling_obj = SMOTE(
                    sampling_strategy='not majority', random_state=1
                )
            elif resampling_method == 'smoteenn':
                resampling_obj = SMOTEENN(
                    sampling_strategy='not majority', random_state=1
                )
            elif resampling_method == 'smotetomek':
                resampling_obj = SMOTETomek(
                    sampling_strategy='not majority', random_state=1
                )
            else:
                raise ValueError(
                    'Resampling method {} not recognised'.format(resampling_method)
                )

        else:
            if resampling_method == 'no_balancing':
                resampling_obj = None
            elif resampling_method == 'max_sampling':
                resampling_obj = RandomOverSampler(sampling_strategy='not majority')
            elif resampling_method == 'smote':
                resampling_obj = SMOTE(sampling_strategy='not majority')
            elif resampling_method == 'smoteenn':
                resampling_obj = SMOTEENN(sampling_strategy='not majority')
            elif resampling_method == 'smotetomek':
                resampling_obj = SMOTETomek(sampling_strategy='not majority')
            else:
                raise ValueError(
                    'Resampling method {} not recognised'.format(resampling_method)
                )

        return resampling_obj

    def run_randomised_search(
        self, x_train, y_train, train_groups, selected_features, clf, splits,
        const_split, resampling_method, n_components_pca, params,
        scoring_metric, n_iter=None, test=False
    ):
        """
        Randomly picks combinations of hyperparameters from an input grid and
        runs cross-validation to calculate their score using the
        RandomizedSearchCV class from sklearn. The cross-validation pipeline
        consists of: 1) selecting the subset of features to include in the
        analysis; 2) standardisation of the data across each feature (by
        subtracting the median and dividing by the IQR); 3) transforming the
        data to a user-specified number of features with PCA; 4) resampling the
        data if imbalanced; and 5) running the selected classifier with randomly
        selected combinations of input hyperparameter values. Note that
        incompatible combinations of hyperparameters will simply be skipped
        (error_score=np.nan), so there is no need to avoid including e.g.
        hyperparameters that are defined only if other hyperparameters are
        defined as particular values. Returns the results of the randomised
        search.

        Input
        ----------
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - train_groups: Numpy array of group names of training data
        - selected_features: List of features to include
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.svm.LinearSVC('dual'=False)
        - splits: List of train: test splits of the input data (x_train and
        y_train) for cross-validation
        - const_split: Boolean, if set to True and splits is set to None,
        function will generate splits using a fixed random_state (=>
        reproducible results when the function is re-run)
        - resampling_method: Name of the method (string) used to resample the
        data in an imbalanced dataset. Recognised method names are:
        'no_balancing'; 'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to transform the data to
        after fitting the data with PCA. If set to None, PCA will not be
        included in the pipeline.
        - params: Dictionary of hyperparameter values to search.
        Key = hyperparameter name (must match the name of the parameter in the
        selected sklearn ML classifier class); Value = range of values to
        test for that hyperparameter - note that all numerical ranges must be
        supplied as numpy arrays in order to avoid throwing an error with the
        imblearn Pipeline() class
        - scoring_metric: Name of the scoring metric used to measure the
        performance of the fitted classifier. Metric must be a string and be one
        of the scoring metrics for classifiers listed at
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
        - n_iter: The number of hyperparameter combinations to test. If set to
        None, will be set to either 100 or 1/3 of the total number of possible
        combinations of hyperparameter values specified in the params dictionary
        (whichever is larger).
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - random_search: RandomizedSearchCV object fitted to the training data
        """

        # Checks argument values are suitable to run the function. Arguments
        # tested by check_arguments that are not input into
        # run_randomised_search are set to values that will enable the tests to
        # pass (but are otherwise unused in the run_randomised_search function)
        check_arguments(
            func_name='run_randomised_search', x_train=x_train, y_train=y_train,
            train_groups=train_groups, x_test=np.array([]), y_test=np.array([]),
            selected_features=selected_features, splits=splits,
            const_split=const_split, resampling_method=resampling_method,
            n_components_pca=n_components_pca, run='randomsearch',
            fixed_params={}, tuned_params=params,
            train_scoring_metric=scoring_metric,
            test_scoring_funcs={}, n_iter=n_iter,
            cv_folds_inner_loop=5, cv_folds_outer_loop='loocv',
            draw_conf_mat=True, plt_name='', test=test
        )

        # Determines number of iterations to run. If this is not defined by the
        # user, it is set to 1/3 * the total number of possible parameter value
        # combinations, with a lower limit of 100. If the total number of
        # possible parameter value combinations is less than 100, all parameter
        # value combinations are tested.
        num_params_combs = 1
        for val in params.values():
            if isinstance(val, (list, np.ndarray)):
                num_params_combs *= len(val)
        if n_iter is None:
            n_iter = int(num_params_combs*(1/3))
            if n_iter < 100:
                n_iter = 100
        if n_iter > num_params_combs:
            n_iter = num_params_combs

        # Generates resampling object
        resampling_obj = self.conv_resampling_method(
            resampling_method=resampling_method, const_split=const_split,
            test=test
        )
        # Generates feature selection object
        if type(selected_features) == int:
            feature_selection = SelectKBest(
                score_func=f_classif, k=selected_features
            )
        else:
            feature_selection = ManualFeatureSelection(
                all_features=self.features, selected_features=selected_features
            )
        # Generates robust scaler object
        standardisation = RobustScaler()

        # Runs randomised search pipeline. Resampling should be run last since
        # its performance will be affected by data preprocessing.
        if n_components_pca is None:
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])
        else:
            pca = PCA(n_components=n_components_pca)
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('PCA', pca),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])

        params = OrderedDict({'{}__{}'.format(type(clf).__name__, key): val
                              for key, val in params.items()})

        if test is False:
            random_search = RandomizedSearchCV(
                estimator=std_pca_clf, param_distributions=params,
                n_iter=n_iter, scoring=scoring_metric, n_jobs=-1, cv=splits,
                error_score=np.nan
            )
        else:
            random_search = RandomizedSearchCV(
                estimator=std_pca_clf, param_distributions=params,
                n_iter=n_iter, scoring=scoring_metric, n_jobs=-1, cv=splits,
                error_score=np.nan, random_state=1
            )
        random_search.fit(X=x_train, y=y_train, groups=train_groups)

        self.flag_extreme_params(random_search.best_params_, params, test=False)

        return random_search

    def run_grid_search(
        self, x_train, y_train, train_groups, selected_features, clf, splits,
        const_split, resampling_method, n_components_pca, params,
        scoring_metric, test=False
    ):
        """
        Tests all possible combinations of hyperparameters from an input grid
        and runs cross-validation to calculate their score using the
        GridSearchCV class from sklearn. The cross-validation pipeline consists
        of: 1) selecting the subset of features to include in the analysis; 2)
        standardisation of the data across each feature (by subtracting the
        median and dividing by the IQR); 3) transforming the data to a
        user-specified number of features with PCA; 4) resampling the data if
        imbalanced; and 5) running the selected classifier with randomly
        selected combinations of input hyperparameter values. Note that
        incompatible combinations of hyperparameters will simply be skipped
        (error_score=np.nan), so there is no need to avoid including e.g.
        hyperparameters that are defined only if other hyperparameters are
        defined as particular values. Returns the results of the grid search.

        Input
        ----------
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - train_groups: Numpy array of group names of training data
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.svm.LinearSVC('dual'=False)
        - splits: List of train: test splits of the input data (x_train and
        y_train) for cross-validation
        - const_split: Boolean, if set to True and splits is set to None,
        function will generate splits using a fixed random_state (=>
        reproducible results when the function is re-run)
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to transform the data to after
        fitting the data with PCA. If set to None, PCA will not be included in
        the pipeline.
        - params: Dictionary of hyperparameter values to search.
        Key = hyperparameter name (must match the name of the parameter in the
        selected sklearn ML classifier class); Value = range of values to
        test for that hyperparameter - note that all numerical ranges must be
        supplied as numpy arrays in order to avoid throwing an error with the
        imblearn Pipeline() class
        - scoring_metric: Name of the scoring metric used to measure the
        performance of the fitted classifier. Metric must be a string and be one
        of the scoring metrics for classifiers listed at
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - grid_search: GridSearchCV object fitted to the training data
        """

        # Checks argument values are suitable to run the function. Arguments
        # tested by check_arguments that are not input into run_grid_search are
        # set to values that will enable the tests to pass (but are otherwise
        # unused in the run_grid_search function)
        check_arguments(
            func_name='run_grid_search', x_train=x_train, y_train=y_train,
            train_groups=train_groups, x_test=np.array([]), y_test=np.array([]),
            selected_features=selected_features,
            splits=splits, const_split=const_split,
            resampling_method=resampling_method,
            n_components_pca=n_components_pca, run='gridsearch',
            fixed_params={}, tuned_params=params,
            train_scoring_metric=scoring_metric,
            test_scoring_funcs={}, n_iter=None,
            cv_folds_inner_loop=5, cv_folds_outer_loop='loocv',
            draw_conf_mat=True, plt_name='', test=test
        )

        # Generates resampling object
        resampling_obj = self.conv_resampling_method(
            resampling_method=resampling_method, const_split=const_split,
            test=test
        )
        # Generates feature selection object
        if type(selected_features) == int:
            feature_selection = SelectKBest(
                score_func=f_classif, k=selected_features
            )
        else:
            feature_selection = ManualFeatureSelection(
                all_features=self.features, selected_features=selected_features
            )
        # Generates robust scaler object
        standardisation = RobustScaler()

        # Runs grid search pipeline. Resampling should be run last since its
        # performance will be affected by data preprocessing
        if n_components_pca is None:
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])
        else:
            pca = PCA(n_components=n_components_pca)
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('PCA', pca),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])

        params = OrderedDict({'{}__{}'.format(type(clf).__name__, key): val
                              for key, val in params.items()})

        grid_search = GridSearchCV(
            estimator=std_pca_clf, param_grid=params, scoring=scoring_metric,
            n_jobs=-1, cv=splits, error_score=np.nan
        )
        grid_search.fit(X=x_train, y=y_train, groups=train_groups)

        self.flag_extreme_params(grid_search.best_params_, params, test=False)

        return grid_search

    def train_model(
        self, x_train, y_train, selected_features, clf, const_split,
        resampling_method, n_components_pca, test=False
    ):
        """
        Trains user-specified model on the training data (without cross-
        validation). Training data is first filtered to retain only the selected
        features, next it is standardised (by subtracting the median and
        dividing by the IQR), then transformed to a user-specified number of
        features with PCA (this step is skipped if n_components_pca is set to
        None), and finally resampled if necessary to balance the class sizes,
        before it is fed into the model for training.

        Input
        ----------
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - selected_features: List of features to include
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.svm.LinearSVC('dual'=False)
        - const_split: Boolean, if set to True and splits is set to None,
        function will generate splits using a fixed random_state (=>
        reproducible results when the function is re-run)
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to transform the data to
        after fitting the data with PCA. If set to None, PCA will not be
        included in the pipeline.
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - std_pca_clf: Model fitted to the training data. This can be fed into
        test_model function to measure how well the model predicts the classes
        of the test data set aside by split_train_test_data.
        """

        # Checks argument values are suitable to run the function. Arguments
        # tested by check_arguments that are not input into train_model are set
        # to values that will enable the tests to pass (but are otherwise unused
        # in the train_model function)
        check_arguments(
            func_name='train_model', x_train=x_train, y_train=y_train,
            train_groups=None, x_test=np.array([]), y_test=np.array([]),
            selected_features=selected_features,
            splits=[(y_train, np.array([]))], const_split=const_split,
            resampling_method=resampling_method,
            n_components_pca=n_components_pca, run='train', fixed_params={},
            tuned_params={}, train_scoring_metric='accuracy',
            test_scoring_funcs={}, n_iter=None, cv_folds_inner_loop=5,
            cv_folds_outer_loop='loocv', draw_conf_mat=True, plt_name='',
            test=test
        )

        # Generates resampling object
        resampling_obj = self.conv_resampling_method(
            resampling_method=resampling_method, const_split=const_split,
            test=test
        )
        # Generates feature selection object
        if type(selected_features) == int:
            feature_selection = SelectKBest(
                score_func=f_classif, k=selected_features
            )
        else:
            feature_selection = ManualFeatureSelection(
                all_features=self.features, selected_features=selected_features
            )
        # Generates robust scaler object
        standardisation = RobustScaler()

        # Runs model training pipeline. Resampling should be run last since its
        # performance will be affected by data preprocessing
        if n_components_pca is None:
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])
        else:
            pca = PCA(n_components=n_components_pca)
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('PCA', pca),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])

        # Fits single model on all training data
        # (No need for train_groups as only required for generating CV splits.)
        std_pca_clf.fit(X=x_train, y=y_train)

        return std_pca_clf

    def test_model(
        self, x_test, y_test, clf, test_scoring_funcs, draw_conf_mat=True,
        plt_name='', test=False
    ):
        """
        Tests model (previously fitted to the training data using e.g.
        train_model function) by predicting the class labels of test data.
        Scores the model by comparing the predicted and actual class labels
        across a range of user-specified scoring functions, plus plots confusion
        matrices.

        Input
        ----------
        - x_test: Numpy array of x values of test data
        - y_test: Numpy array of y values of test data
        - clf: Model previously fitted to the training data
        - test_scoring_funcs: Dictionary of sklearn scoring functions and
        dictionaries of arguments to be fed into these functions,
        e.g. {sklearn.metrics.f1_score: {'average': 'macro'}}
        - draw_conf_mat: Boolean, dictates whether to plot confusion matrices to
        compare the model predictions to the test data. By default set to True.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - predictions: Numpy array of the predicted class values made by the
        trained classifier (clf)
        - test_scores: Dictionary of user-specified scoring functions and their
        values as calculated from the class label predictions made by the model
        on the testing data
        """

        # Checks argument values are suitable to run the function. Arguments
        # tested by check_arguments that are not input into test_model are set
        # to values that will enable the tests to pass (but are otherwise unused
        # in the test_model function)
        check_arguments(
            func_name='test_model', x_train=np.array([]), y_train=np.array([]),
            train_groups=None, x_test=x_test, y_test=y_test,
            selected_features=[], splits=[(np.array([]), np.array([]))],
            const_split=True, resampling_method='no_balancing',
            n_components_pca=None, run='randomsearch', fixed_params={},
            tuned_params={}, train_scoring_metric='accuracy',
            test_scoring_funcs=test_scoring_funcs, n_iter=None,
            cv_folds_inner_loop=5, cv_folds_outer_loop='loocv',
            draw_conf_mat=draw_conf_mat, plt_name=plt_name, test=test
        )

        # Uses trained model to predict classes of test data. Note that feature
        # selection, scaling, PCA and rebalancing will be performed within the
        # pipeline on the test data, so don't need to do this manually
        # beforehand
        predictions = clf.predict(X=x_test)

        # Calculates selected scoring metrics
        test_scores = OrderedDict()
        for func, params in test_scoring_funcs.items():
            if func.__name__ == 'cohen_kappa_score':
                params['y1'] = y_test
                params['y2'] = predictions
            elif func.__name__ == 'roc_auc_score':
                params['y_true'] = y_test
                params['y_score'] = predictions
            else:
                params['y_true'] = y_test
                params['y_pred'] = predictions
            test_score = func(**params)
            test_scores[func.__name__.replace('_score', '')] = test_score

        # Generates confusion matrices
        if draw_conf_mat is True:
            plt_name = '{}{}'.format(plt_name, type(clf).__name__)
            draw_conf_matrices(
                y_test, predictions, self.results_dir, plt_name, test
            )

        return predictions, test_scores

    def run_ml(
        self, clf, x_train, y_train, train_groups, x_test, y_test,
        selected_features, splits, const_split, resampling_method,
        n_components_pca, run, params, train_scoring_metric,
        test_scoring_funcs={}, n_iter=None, cv_folds=5, draw_conf_mat=True,
        plt_name='', test=False
    ):
        """
        Function to either test a range of parameter combinations with
        RandomizedSearchCV or GridSearchCV for a particular model, or to train
        and test a model using a predefined set of parameters.

        Input
        ----------
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.svm.LinearSVC('dual'=False) (for run='randomsearch' or
        run='gridsearch'), or sklearn.svm.SVC (for run='train')
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - train_groups: Numpy array of group names of training data
        - x_test: Numpy array of x values of test data
        - y_test: Numpy array of y values of test data
        - selected_features: List of features to include
        - splits: Either a list of train: test splits of the input data
        (x_train and y_train) for cross-validation, or set to None (in which
        case run_ml runs stratified k-folds/group k-folds in sklearn to
        construct the list, as directed by "cv_folds")
        - const_split: Boolean, if set to True and splits is set to None,
        function will generate splits using a fixed random_state (=>
        reproducible results when the function is re-run)
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to transform the data to
        after fitting the data with PCA. If set to None, PCA will not be
        included in the pipeline.
        - run: Either 'randomsearch', 'gridsearch' or 'train'. Directs the
        function whether to run cross-validation with RandomizedSearchCV or
        GridSearchCV to select a suitable combination of hyperparameter values,
        or whether to train and test the model.
        - params: If run == 'randomsearch' or 'gridsearch', params is a
        dictionary of hyperparameter values to search. Key = hyperparameter name
        (must match the name of the parameter in the selected sklearn ML
        classifier class); Value = range of values to test for that
        hyperparameter - note that all numerical ranges must be supplied as
        numpy arrays in order to avoid throwing an error with the imblearn
        Pipeline() class. Else if run == 'train', params is also a dictionary of
        hyperparameters, but in this case a single value is provided for each
        hyperparameter as opposed to a range.
        - train_scoring_metric: Name of the scoring metric used to measure the
        performance of the fitted classifier. Metric must be a string and be one
        of the scoring metrics for classifiers listed at
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
        - test_scoring_funcs: Dictionary of sklearn scoring functions and
        dictionaries of arguments to be fed into these functions.
        E.g. {sklearn.metrics.f1_score: {'average': 'macro'}}
        - n_iter: Integer number of hyperparameter combinations to test / ''. If
        set to '' will be set to either 100 or 1/3 of the total number of
        possible combinations of hyperparameter values specified in the params
        dictionary (whichever is larger).
        - cv_folds: Integer number of folds to run in cross-validation. E.g. as
        a default cv_folds = 5, which generates 5 folds of training and
        validation data, with 80% and 20% of the data forming the training and
        validation sets respectively.
        - draw_conf_mat: Boolean, dictates whether to plot confusion matrices to
        compare the model predictions to the test data. By default set to True.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - search: Dictionary of the selected resampling methods and the
        corresponding output from fitting the user-specified algorithm (either a
        RandomizedSearchCV / GridSearchCV object, or a model fitted to the
        resampled training data).
        - trained_clf: Model fitted to the training data
        - predictions: Numpy array of the predicted class values made by the
        trained classifier
        - test_scores: Dictionary of user-specified scoring functions and their
        values as calculated from the class label predictions made by the model
        on the testing data
        """

        # Checks argument values are suitable to run the function. Arguments
        # tested by check_arguments that are not input into run_ml are set to
        # values that will enable the tests to pass (but are otherwise unused in
        # the run_ml function)
        if splits is None:
            check_splits = [(y_train, np.array([]))]
        else:
            check_splits = splits
        check_arguments(
            func_name='run_ml', x_train=x_train, y_train=y_train,
            train_groups=train_groups, x_test=x_test, y_test=y_test,
            selected_features=selected_features, splits=check_splits,
            const_split=const_split, resampling_method=resampling_method,
            n_components_pca=n_components_pca, run=run, fixed_params={},
            tuned_params=params, train_scoring_metric=train_scoring_metric,
            test_scoring_funcs=test_scoring_funcs, n_iter=n_iter,
            cv_folds_inner_loop=cv_folds, cv_folds_outer_loop='loocv',
            draw_conf_mat=draw_conf_mat, plt_name=plt_name, test=test
        )

        if splits is None and run in ['randomsearch', 'gridsearch']:
            skf = None
            gkf = None

            # There must be more than cv_folds instances of each dataset
            if cv_folds > x_train.shape[0]:
                raise ValueError(
                    'The number of k-folds must be smaller than the number of '
                    'data points in the training dataset'
                )

            if train_groups is None:
                if test is True or const_split is True:
                    skf = StratifiedKFold(
                        n_splits=cv_folds, shuffle=True, random_state=1
                    )
                else:
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True)
                splits = list(skf.split(X=x_train, y=y_train))
            else:
                gkf = GroupKFold(n_splits=cv_folds)
                splits = list(gkf.split(X=x_train, y=y_train, groups=train_groups))

        if run == 'randomsearch':
            search = self.run_randomised_search(
                x_train, y_train, train_groups, selected_features, clf, splits,
                const_split, resampling_method, n_components_pca, params,
                train_scoring_metric, n_iter, test
            )
            return search

        elif run == 'gridsearch':
            search = self.run_grid_search(
                x_train, y_train, train_groups, selected_features, clf, splits,
                const_split, resampling_method, n_components_pca, params,
                train_scoring_metric, test
            )
            return search

        elif run == 'train':
            train_clf = clf(**params)
            train_clf = self.train_model(
                x_train, y_train, selected_features, train_clf, const_split,
                resampling_method, n_components_pca, test
            )
            predictions, test_scores = self.test_model(
                x_test, y_test, train_clf, test_scoring_funcs, draw_conf_mat,
                plt_name, test
            )
            return train_clf, predictions, test_scores

    def run_nested_CV(
        self, clf, x, y, groups, selected_features, inner_splits, outer_splits,
        const_split, resampling_method, n_components_pca, run, fixed_params,
        tuned_params, train_scoring_metric, test_scoring_funcs, n_iter=None,
        cv_folds_inner_loop=5, cv_folds_outer_loop='loocv', draw_conf_mat=False,
        plt_name='', test=False
    ):
        """
        Runs nested cross-validation to fit an input sklearn classifier to the
        data

        Input
        ----------
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.svm.LinearSVC
        - x: Numpy array of all x data (no need to have already split into
        training and test data)
        - y: Numpy array of all y data (no need to have already split into
        training and test data)
        - groups: Numpy array of all group names (no need to have already split
        into training and test data)
        - selected_features: List of features to include
        - inner_splits: Either a list of train: test splits or set to None. Is
        provided to the run_ml function called by run_nested_CV as its "splits"
        parameter.
        - outer_splits: Either a list of train: test splits of the input data (x
        and y) for cross-validation, or set to None (in which case run_nested_CV
        runs leave-one-out cross-validation/stratified k-folds/group k-folds in
        sklearn to construct the generator)
        - const_split: Boolean, if set to True and splits is set to None,
        function will generate splits using a fixed random_state (=>
        reproducible results when the function is re-run)
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to transform the data to
        after fitting the data with PCA. If set to None, PCA will not be
        included in the pipeline.
        - run: Either 'randomsearch' or 'gridsearch'. Directs the function
        whether to run inner loop cross-validation with RandomizedSearchCV or
        GridSearchCV to select a suitable combination of hyperparameter values.
        - fixed_params: Dictionary of fixed-value hyperparameters and their
        selected values, e.g. {n_jobs: -1}
        - tuned_params: Dictionary of hyperparameter values to search.
        Key = hyperparameter name (must match the name of the parameter in the
        selected clf class); Value = range of values to test for that
        hyperparameter - note that all numerical ranges must be supplied as
        numpy arrays in order to avoid throwing an error with the imblearn
        Pipeline() class.
        - train_scoring_metric: Name of the scoring metric used to measure the
        performance of the fitted classifier. Metric must be a string and be one
        of the scoring metrics for classifiers listed at
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
        - test_scoring_funcs: Dictionary of sklearn scoring functions and
        dictionaries of arguments to be fed into these functions,
        e.g. {sklearn.metrics.f1_score: {'average': 'macro'}}, that are used to
        measure the performance of the trained classifier on a holdout dataset
        - n_iter: Required if run == 'randomsearch'. Integer number of
        hyperparameter combinations to test / None. If None, will be set to
        either 100 or 1/3 of the total number of possible combinations of
        hyperparameter values specified in the params dictionary (whichever is
        larger).
        - cv_folds_inner_loop: Integer number of folds to run in the inner
        cross-validation loop. E.g. as a default cv_folds_inner_loop = 5, which
        generates 5 folds of training and validation data, with 80% and 20% of
        the data forming the training and validation sets respectively.
        - cv_folds_outer_loop: Integer number of folds to run in the outer
        cross-validation loop. Alternatively, set to 'loocv' (default value),
        which sets the number of folds equal to the size of the dataset
        (leave-one-out cross-validation).
        - draw_conf_mat: Boolean, dictates whether to plot confusion matrices to
        compare the model predictions to the test data. By default set to False.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - nested_cv_search: Dictionary of trained models and the values of the
        metrics included in test_score_funcs, for each split in the outer loop
        of nested cross-validation
        """

        # Checks argument values are suitable to run the function. Arguments
        # tested by check_arguments that are not input into run_nested_CV are
        # set to values that will enable the tests to pass (but are otherwise
        # unused in the run_nested_CV function)
        if outer_splits is None:  # Inner_splits is checked in the run_ml
        # function called below
            check_splits = [(y, np.array([]))]
        else:
            check_splits = outer_splits
        check_arguments(
            func_name='run_nested_CV', x_train=x, y_train=y,
            train_groups=groups, x_test=np.array([]), y_test=np.array([]),
            selected_features=selected_features, splits=check_splits,
            const_split=const_split, resampling_method=resampling_method,
            n_components_pca=n_components_pca, run=run,
            fixed_params=fixed_params, tuned_params=tuned_params,
            train_scoring_metric=train_scoring_metric,
            test_scoring_funcs=test_scoring_funcs, n_iter=n_iter,
            cv_folds_inner_loop=cv_folds_inner_loop,
            cv_folds_outer_loop=cv_folds_outer_loop,
            draw_conf_mat=draw_conf_mat, plt_name=plt_name, test=test
        )

        if not inner_splits is None:
            if not type(inner_splits) in [dict, OrderedDict]:
                raise TypeError(
                    'Expect "inner_splits" to be set either to None, or to a '
                    'dictionary of train:test splits for the inner '
                    'cross-validation loop'
                )
            else:
                if outer_splits is None:
                    if type(cv_folds_outer_loop) != int:
                        raise ValueError(
                           'Mismatch in the number of train:test splits in the'
                           ' outer CV loop and the number of entries in the '
                           '"inner_splits" dictionary'
                        )
                    else:
                        if cv_folds_outer_loop != len(inner_splits):
                            raise ValueError(
                                'Mismatch in the number of train:test splits in'
                                ' the outer CV loop and the number of entries '
                                'in the "inner_splits" dictionary'
                            )
                else:
                    if len(outer_splits) != len(inner_splits):
                        raise ValueError(
                            'Mismatch in the number of train:test splits in the'
                            ' outer CV loop and the number of entries in the '
                            '"inner_splits" dictionary'
                        )

        nested_cv_search = OrderedDict({
            'inner_loop_searches': [],
            'outer_loop_models': [],
            'outer_loop_params': [],
            'test_scores': OrderedDict(),
            'predictions': [],
            'x_true': [],
            'y_true': [],
            'groups_true': []
        })
        for scoring_func in test_scoring_funcs.keys():
            nested_cv_search['test_scores'][
                scoring_func.__name__.replace('_score', '')
            ] = []

        # Split data into train and test sets
        if outer_splits is None:
            loocv = None
            logo = None
            skf = None
            gkf = None
            if cv_folds_outer_loop == 'loocv':
                if groups is None:
                    loocv = LeaveOneOut()
                    outer_splits = list(loocv.split(X=x, y=y))
                else:
                    logo = LeaveOneGroupOut()
                    outer_splits = list(logo.split(X=x, y=y, groups=groups))

            else:
                # There must be more than cv_folds_outer_loop instances of the
                # dataset
                if cv_folds_outer_loop > x.shape[0]:
                    raise ValueError(
                        'The number of k-folds must be smaller than the number '
                        'of data points'
                    )
                # Generates splits
                if groups is None:
                    if test is True or const_split is True:
                        skf = StratifiedKFold(
                            n_splits=cv_folds_outer_loop, shuffle=True,
                            random_state=1
                        )
                    else:
                        skf = StratifiedKFold(
                            n_splits=cv_folds_outer_loop, shuffle=True
                        )
                    outer_splits = list(skf.split(X=x, y=y))
                else:
                    gkf = GroupKFold(n_splits=cv_folds_outer_loop)
                    outer_splits = list(gkf.split(X=x, y=y, groups=groups))

        for i, split in enumerate(outer_splits):
            train_split = split[0]
            test_split = split[1]
            x_train = copy.deepcopy(x)[train_split]
            y_train = copy.deepcopy(y)[train_split]
            x_test = copy.deepcopy(x)[test_split]
            y_test = copy.deepcopy(y)[test_split]
            if not groups is None:
                train_groups = copy.deepcopy(groups)[train_split]
                test_groups = copy.deepcopy(groups)[test_split]
            else:
                train_groups = None
                test_groups = None

            if not inner_splits is None:
                sub_inner_split = inner_splits[i]
            else:
                sub_inner_split = None

            # Random / grid search of hyperparameters on training set
            train_clf = clf(**fixed_params)
            search = self.run_ml(
                train_clf, x_train, y_train, train_groups, np.array([]),
                np.array([]), selected_features, sub_inner_split, const_split,
                resampling_method, n_components_pca, run, tuned_params,
                train_scoring_metric, {}, n_iter, cv_folds_inner_loop,
                draw_conf_mat, '', test
            )
            nested_cv_search['inner_loop_searches'].append(search)

            # Train clf with best hyperparameter selection on entire training
            # split, then make predictions and calculate selected statistics on
            # test split
            best_params = OrderedDict({
                key.split('__')[1]: val for key, val in search.best_params_.items()
            })
            best_params = OrderedDict({**fixed_params, **best_params})
            fold_search, fold_predictions, fold_test_scores = self.run_ml(
                clf, x_train, y_train, train_groups, x_test, y_test,
                selected_features, None, const_split, resampling_method,
                n_components_pca, 'train', best_params, train_scoring_metric,
                test_scoring_funcs, None, cv_folds_inner_loop, draw_conf_mat,
                plt_name, test
            )

            nested_cv_search['outer_loop_params'].append(best_params)
            nested_cv_search['outer_loop_models'].append(fold_search)
            for scoring_func_name, val in fold_test_scores.items():
                nested_cv_search['test_scores'][scoring_func_name].append(val)
            nested_cv_search['predictions'].append(fold_predictions)
            nested_cv_search['x_true'].append(x_test)
            nested_cv_search['y_true'].append(y_test)
            nested_cv_search['groups_true'].append(test_groups)

        # Checks nested_cv_search dictionary has been updated with the expected
        # values
        for key, val in nested_cv_search.items():
            if key == 'test_scores':
                for sub_key, sub_val in nested_cv_search[key].items():
                    if len(sub_val) != len(outer_splits):
                        raise Exception(
                            'Output dictionary from running nested '
                            'cross-validation does not contain the expected '
                            'number of entries'
                        )
            else:
                if len(val) != len(outer_splits):
                    raise Exception(
                        'Output dictionary from running nested cross-validation'
                        ' does not contain the expected number of entries'
                    )

        # Calculate average and standard deviation across cv_folds_outer_loop
        # folds. Have not included 2.5th and 97.5th percentiles as highly
        # unlikely that there will be sufficient CV folds to calculate these
        # metrics accurately
        nested_cv_search['average_test_scores'] = OrderedDict()
        for scoring_func_name, score_list in nested_cv_search['test_scores'].items():
            nested_cv_search['average_test_scores'][scoring_func_name] = np.mean(score_list)
        nested_cv_search['stddev_test_scores'] = OrderedDict()
        for scoring_func_name, score_list in nested_cv_search['test_scores'].items():
            nested_cv_search['stddev_test_scores'][scoring_func_name] = np.std(score_list, ddof=1)  # Sample standard deviation

        # Records which outer loop parameters lead to the best model performance
        nested_cv_search['best_outer_loop_params'] = OrderedDict()
        for scoring_func_name in nested_cv_search['test_scores'].keys():
            best_index = np.where(
                   nested_cv_search['test_scores'][scoring_func_name]
                == np.amax(nested_cv_search['test_scores'][scoring_func_name])
            )[0][0]
            nested_cv_search['best_outer_loop_params'][scoring_func_name] = (
                nested_cv_search['outer_loop_params'][best_index]
            )

        return nested_cv_search

    def run_5x2_CV_combined_F_test(
        self, x, y, selected_features_1, selected_features_2,
        classifier_1, classifier_2, params_1, params_2, resampling_method_1,
        resampling_method_2, n_components_pca_1, n_components_pca_2,
        test=False
    ):
        """
        Runs 5x2 CV combined F test to calculate whether there is a significant
        difference in performance between two classifier models. In this case,
        train/test splits are made using GroupKFold (from the sklearn package).
        These models can differ in: 1) the classifier algorithm selected; 2) the
        values of the classifier parameter values; 3) the features selected in
        the input data; 4) the number of PCA components used to transform the
        input data; or 5) the method used to resample imbalanced classes in the
        input dataset.

        Input
        ----------
        - x: Numpy array of all x data (no need to have already split into
        training and test data)
        - y: Numpy array of all y data (no need to have already split into
        training and test data)
        - selected_features_1: List of features to include when training
        classifier_1
        - selected_features_2: List of features to include when training
        classifier_2
        - classifier_1: Selected classifier from the sklearn package,
        e.g. sklearn.svm.LinearSVC, whose performance on the input data is to be
        compared to classifier_2
        - classifier_2: Selected classifier from the sklearn package,
        e.g. sklearn.ensemble.AdaBoost, whose performance on the input data is
        to be compared to classifier_1
        - params_1: Dictionary of fixed-value hyperparameters and their
        selected values, e.g. {n_jobs: -1}, to be used for classifier 1
        - params_2: Dictionary of fixed-value hyperparameters and their
        selected values, e.g. {n_jobs: -1}, to be used for classifier 2
        - resampling_method_1: Name of the method used to resample the data in
        the classifier 1 pipeline. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - resampling_method_2: Name of the method used to resample the data in
        the classifier 2 pipeline. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca_1: The number of components to transform the data to
        after fitting the data with PCA for classifier 1. If set to None, PCA
        will not be included in the pipeline.
        - n_components_pca_2: The number of components to transform the data to
        after fitting the data with PCA for classifier 2. If set to None, PCA
        will not be included in the pipeline.
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - F: F statistic output from combined F-test
        - p: p value output from combined F-test
        """

        f1 = make_scorer(f1_score, average='weighted')

        # Checks argument values are suitable to run the function. Arguments of
        # "check_arguments" that are not specified for
        # "run_5x2_CV_combined_F_test" are set to values that will pass the checks
        check_arguments(
            func_name='run_5x2_CV_combined_F_test', x_train=x, y_train=y,
            train_groups=None, x_test=np.array([]), y_test=np.array([]),
            selected_features=selected_features_1,
            splits=[(y, np.array([]))], const_split=False,
            resampling_method=resampling_method_1,
            n_components_pca=n_components_pca_1, run='randomsearch',
            fixed_params=params_1, tuned_params={},
            train_scoring_metric=f1, test_scoring_funcs={}, n_iter=None,
            cv_folds_inner_loop=5, cv_folds_outer_loop='loocv',
            draw_conf_mat=True, plt_name='', test=test
        )
        check_arguments(
            func_name='run_5x2_CV_combined_F_test', x_train=x, y_train=y,
            train_groups=None, x_test=np.array([]), y_test=np.array([]),
            selected_features=selected_features_2,
            splits=[(y, np.array([]))], const_split=False,
            resampling_method=resampling_method_2,
            n_components_pca=n_components_pca_2, run='randomsearch',
            fixed_params=params_2, tuned_params={},
            train_scoring_metric=f1, test_scoring_funcs={}, n_iter=None,
            cv_folds_inner_loop=5, cv_folds_outer_loop='loocv',
            draw_conf_mat=True, plt_name='', test=test
        )

        clf_1 = classifier_1(**params_1)
        trained_clf_1 = self.train_model(
            x, y, selected_features_1, clf_1, False, resampling_method_1,
            n_components_pca_1, test
        )
        clf_2 = classifier_2(**params_2)
        trained_clf_2 = self.train_model(
            x, y, selected_features_2, clf_2, False, resampling_method_2,
            n_components_pca_2, test
        )

        if test is False:
            F, p = combined_ftest_5x2cv(
                estimator1=trained_clf_1, estimator2=trained_clf_2, X=x, y=y,
                scoring=f1
            )
        else:
            F, p = combined_ftest_5x2cv(
                estimator1=trained_clf_1, estimator2=trained_clf_2, X=x, y=y,
                scoring=f1, random_seed=1
            )

        return F, p

    def run_group_5x2_CV_combined_F_test(
        self, x, y, groups, selected_features_1, selected_features_2,
        classifier_1, classifier_2, params_1, params_2, resampling_method_1,
        resampling_method_2, n_components_pca_1, n_components_pca_2, test=False
    ):
        """
        Runs 5x2 CV combined F test to calculate whether there is a significant
        difference in performance between two classifier models. In this case,
        train/test splits are made using GroupKFold (from the sklearn package).
        These models can differ in: 1) the classifier algorithm selected; 2) the
        values of the classifier parameter values; 3) the features selected in
        the input data; 4) the number of PCA components used to transform the
        input data; or 5) the method used to resample imbalanced classes in the
        input dataset.

        Input
        ----------
        - x: Numpy array of all x data (no need to have already split into
        training and test data)
        - y: Numpy array of all y data (no need to have already split into
        training and test data)
        - groups: Numpy array of all group names (no need to have already split
        into training and test data)
        - selected_features_1: List of features to include when training
        classifier_1
        - selected_features_2: List of features to include when training
        classifier_2
        - classifier_1: Selected classifier from the sklearn package,
        e.g. sklearn.svm.LinearSVC, whose performance on the input data is to be
        compared to classifier_2
        - classifier_2: Selected classifier from the sklearn package,
        e.g. sklearn.ensemble.AdaBoost, whose performance on the input data is
        to be compared to classifier_1
        - params_1: Dictionary of fixed-value hyperparameters and their
        selected values, e.g. {n_jobs: -1}, to be used for classifier 1
        - params_2: Dictionary of fixed-value hyperparameters and their
        selected values, e.g. {n_jobs: -1}, to be used for classifier 2
        - resampling_method_1: Name of the method used to resample the data in
        the classifier 1 pipeline. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - resampling_method_2: Name of the method used to resample the data in
        the classifier 2 pipeline. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca_1: The number of components to transform the data to
        after fitting the data with PCA for classifier 1. If set to None, PCA
        will not be included in the pipeline.
        - n_components_pca_2: The number of components to transform the data to
        after fitting the data with PCA for classifier 2. If set to None, PCA
        will not be included in the pipeline.
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - F: F statistic output from combined F-test
        - p: p value output from combined F-test
        """

        check_arguments(
            func_name='run_5x2_CV_combined_F_test', x_train=x, y_train=y,
            train_groups=groups, x_test=np.array([]), y_test=np.array([]),
            selected_features=selected_features_1,
            splits=[(y, np.array([]))], const_split=False,
            resampling_method=resampling_method_1,
            n_components_pca=n_components_pca_1, run='randomsearch',
            fixed_params=params_1, tuned_params={},
            train_scoring_metric='accuracy', test_scoring_funcs={},
            n_iter=None, cv_folds_inner_loop=5, cv_folds_outer_loop='loocv',
            draw_conf_mat=True, plt_name='', test=test
        )
        check_arguments(
            func_name='run_5x2_CV_combined_F_test', x_train=x, y_train=y,
            train_groups=groups, x_test=np.array([]), y_test=np.array([]),
            selected_features=selected_features_2,
            splits=[(y, np.array([]))], const_split=False,
            resampling_method=resampling_method_2,
            n_components_pca=n_components_pca_2, run='randomsearch',
            fixed_params=params_2, tuned_params={},
            train_scoring_metric='accuracy', test_scoring_funcs={},
            n_iter=None, cv_folds_inner_loop=5, cv_folds_outer_loop='loocv',
            draw_conf_mat=True, plt_name='', test=test
        )

        clf_1 = classifier_1(**params_1)
        trained_clf_1 = self.train_model(
            x, y, selected_features_1, clf_1, False, resampling_method_1,
            n_components_pca_1, test
        )
        clf_2 = classifier_2(**params_2)
        trained_clf_2 = self.train_model(
            x, y, selected_features_2, clf_2, False, resampling_method_2,
            n_components_pca_2, test
        )

        variances = []
        differences = []
        for i in range(5):
            all_x = copy.deepcopy(x)
            all_y = copy.deepcopy(y)
            all_y.shape = (all_y.shape[0], 1)
            all_groups = copy.deepcopy(groups)
            all_groups.shape = (all_groups.shape[0], 1)

            merged_data = np.concatenate([all_x, all_y, all_groups], axis=1)
            if test is False:
                np.random.shuffle(merged_data)
            all_x = merged_data[:,:-2]
            all_y = merged_data[:,-2]
            all_groups = merged_data[:,-1]

            splits = list(GroupKFold(n_splits=2).split(
                X=all_x, y=all_y, groups=all_groups
            ))

            sub_diffs = []
            for split in splits:
                x_train = all_x[split[0]]
                x_test = all_x[split[1]]
                y_train = all_y[split[0]]
                y_test = all_y[split[1]]

                trained_clf_1.fit(x_train, y_train)
                trained_clf_2.fit(x_train, y_train)
                predictions_1 = trained_clf_1.predict(x_test)
                predictions_2 = trained_clf_2.predict(x_test)

                score_1 = f1_score(
                    y_true=y_test, y_pred=predictions_1, average='weighted'
                )
                score_2 = f1_score(
                    y_true=y_test, y_pred=predictions_2, average='weighted'
                )
                diff = score_1 - score_2
                sub_diffs.append(diff)

            differences += [sub_diffs[0]**2, sub_diffs[1]**2]
            score_mean = np.mean(sub_diffs)
            score_var = ((sub_diffs[0] - score_mean)**2 + (sub_diffs[1] - score_mean)**2)
            variances.append(score_var)

        numerator = np.sum(differences)
        denominator = 2*(np.sum(variances))
        F = float(numerator / denominator)
        p = float(f.sf(F, 10, 5))

        return F, p
