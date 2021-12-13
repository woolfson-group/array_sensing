
# python -m unittest tests/test_ml_training.py

import numpy as np
import pandas as pd
import unittest
from collections import OrderedDict
from subroutines.exceptions import AlgorithmError, create_generator
from subroutines.train import (
    make_separate_subclass_splits, bootstrap_data, make_feat_importance_plots,
    check_arguments, RunML
)


class TestClass(unittest.TestCase):

    def test_make_separate_subclass_splits(self):
        """
        Tests make_separate_subclass_splits in train.py
        """

        print('Testing make_separate_subclass_splits')

    def test_bootstrap_data(self):
        """
        Tests bootstrap_data in train.py
        """

        print('Testing bootstrap_data')

    def test_make_feat_importance_plots(self):
        """
        Tests make_feat_importance_plots in train.py
        """

        print('Testing make_feat_importance_plots')

    def test_check_arguments(self):
        """
        Tests check_arguments in train.py
        """

        print('Testing check_arguments')

        # Sets "recognised" parameter values that will not raise an exception
        x_train = np.array([])
        y_train = np.array([])
        train_groups = np.array([])
        x_test = np.array([])
        y_test = np.array([])
        selected_features = []
        splits = create_generator(x_train.shape[0])
        resampling_method = 'no_balancing'
        n_components_pca = None
        run = 'randomsearch'
        fixed_params = {}
        tuned_params = {}
        train_scoring_metric = 'accuracy'
        test_scoring_funcs = {}
        n_iter = None
        cv_folds_inner_loop = 5
        cv_folds_outer_loop = 5
        draw_conf_mat = True
        plt_name = ''

        # "Recognised" parameter values should not raise an exception
        output_str = check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(output_str, 'All checks passed')

        # "Unrecognised" parameter values should raise an exception
        # Tests x_train type
        x_train_str = ''
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train_str, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "x_train" to be a numpy array of '
            'training data fluorescence readings'
        )

        # Tests y_train type
        y_train_str = ''
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train_str, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "y_train" to be a numpy array of '
            'training data class labels'
        )

        # Tests train_groups type
        train_groups_str = ''
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups_str, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "train_groups" to be a numpy array '
            'of training data subclass labels'
        )

        # Tests x_test type
        x_test_str = ''
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test_str, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "x_test" to be a numpy array of '
            'test data fluorescence readings'
        )

        # Tests y_test type
        y_test_str = ''
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test_str,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "y_test" to be a numpy array of '
            'test data class labels'
        )

        # Tests y_train is a 1D array
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([[2, 2], [2, 2], [2, 2], [2, 2]])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups, x_test,
            y_test, selected_features, splits, resampling_method,
            n_components_pca, run, fixed_params, tuned_params,
            train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Expect "y_train" to be a 1D array'
        )

        # Tests mismatch in x_train and y_train shape
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2, 2])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups, x_test,
            y_test, selected_features, splits, resampling_method,
            n_components_pca, run, fixed_params, tuned_params,
            train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Different number of entries (rows) in '
            '"x_train" and "y_train"'
        )

        # Tests train_groups is a 1D array
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([[3], [3], [3], [3]])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test, y_test, selected_features, splits, resampling_method,
            n_components_pca, run, fixed_params, tuned_params,
            train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Expect "train_groups" to be a 1D array'
        )

        # Tests mismatch in x_train and train_groups shape
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test, y_test, selected_features, splits, resampling_method,
            n_components_pca, run, fixed_params, tuned_params,
            train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Different number of entries (rows) in '
            '"x_train" and "train_groups"'
        )

        # Tests y_test is a 1D array
        x_test_array = np.array([[4, 4], [4, 4], [4, 4], [4, 4]])
        y_test_array = np.array([[5, 5], [5, 5], [5, 5], [5, 5]])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test_array,
            y_test_array, selected_features, splits, resampling_method,
            n_components_pca, run, fixed_params, tuned_params,
            train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Expect "y_test" to be a 1D array'
        )

        # Tests mismatch in x_test and y_test shape
        x_test_array = np.array([[4, 4], [4, 4], [4, 4], [4, 4]])
        y_test_array = np.array([5, 5, 5, 5, 5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test_array,
            y_test_array, selected_features, splits, resampling_method,
            n_components_pca, run, fixed_params, tuned_params,
            train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Different number of entries (rows) in '
            '"x_test" and "y_test"'
        )

        # Tests mismatch in x_train and x_test shape
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4, 4], [4, 4, 4]])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Different number of features incorporated '
            'in the training and test data'
        )

        # Tests no NaN in x_train
        x_train_array = np.array([[1, np.nan], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'NaN value(s) detected in "x_train" data'
        )

        # Tests no non-numeric entries in x_train
        x_train_array = np.array([[1, 1], [1, 'X'], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Non-numeric value(s) in "x_train" - expect'
            ' all values in "x_train" to be integers / floats'
        )

        # Tests no NaN in y_train
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, np.nan, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'NaN value(s) detected in "y_train" data'
        )

        # Tests no NaN in train_groups
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([np.nan, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'NaN value(s) detected in "train_groups" data'
        )

        # Tests no NaN in x_test
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[np.nan, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'NaN value(s) detected in "x_test" data'
        )

        # Tests no non-numeric values in x_test
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 'X']])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Non-numeric value(s) in "x_test" - expect '
            'all values in "x_test" to be integers / floats'
        )

        # Tests no NaN in y_test
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, np.nan])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'NaN value(s) detected in "y_test" data'
        )

        # Test selected_features is a list
        selected_features_str = 'X'
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features_str, splits, resampling_method, n_components_pca,
            run, fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "selected_features" to be a list of'
            ' features to retain in the analysis'
        )

        # Test length of selected_features list is less than or equal to the
        # number of columns in x_train
        selected_features_list = ['X', 'X', 'X']
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'There is a greater number of features '
            'listed in "selected_features" than there are columns in the '
            '"x_train" input arrays'
        )

        # Test length of selected_features list is less than or equal to the
        # number of columns in x_test (when x_train is not defined)
        selected_features_list = ['X', 'X', 'X']
        x_train_array = np.array([])
        y_train_array = np.array([])
        train_groups_array = None
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'There is a greater number of features '
            'listed in "selected_features" than there are columns in the '
            '"x_test" input arrays'
        )

        # Tests splits type
        splits_list = []
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits_list, resampling_method, n_components_pca,
            run, fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "splits" to be a generator that '
            'creates train test splits'
        )

        # Tests generator creates splits of the expected size
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        selected_features_list = ['X', 'X']
        splits_gen = create_generator(x_train_array.shape[0]+1)
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_gen,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Size of train test splits generated by '
            '"splits" does not match the number of rows in the input array '
            '"x_train"'
        )

        # Tests resampling_method is recognised
        resampling_method_str = ''
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method_str, n_components_pca,
            run, fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), '"resampling_method" unrecognised - expect '
            'value to be one of the following list entries:\n[\'no_balancing\','
            ' \'max_sampling\', \'smote\', \'smoteenn\', \'smotetomek\']'
        )

        # Test n_components_pca is an integer
        n_components_pca_str = 2.0
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca_str,
            run, fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "n_components_pca" to be set either'
            ' to None or to a positive integer value between 1 and the number '
            'of features'
        )

        # Test n_components_pca is an integer in the range of 1 - number of
        # features
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        selected_features_list = ['X', 'X']
        splits_gen = create_generator(x_train_array.shape[0])
        n_components_pca_int = x_train_array.shape[1] + 1
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_gen,
            resampling_method, n_components_pca_int, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Expect "n_components_pca" to be set either'
            ' to None or to a positive integer value between 1 and the number '
            'of features'
        )

        # Tests requirement for run to be "randomsearch", "gridsearch" or
        # "train" when func_name is set to "run_ml"
        x_train_array = np.array([])
        y_train_array = np.array([])
        train_groups_array = np.array([])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        selected_features_list = ['X', 'X']
        splits_gen = create_generator(x_train_array.shape[0])
        n_components_pca_int = x_test_array.shape[1]
        run_str = 'random search'
        with self.assertRaises(ValueError) as message: check_arguments(
            'run_ml', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_gen,
            resampling_method, n_components_pca_int, run_str, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Expect "run" to be set to either '
            '"randomsearch", "gridsearch" or "train"'
        )

        # Tests requirement for run to be "randomsearch" or "gridsearch" when
        # when func_name is set to "run_nested_CV"
        run_str = 'train'
        with self.assertRaises(ValueError) as message: check_arguments(
            'run_nested_CV', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca,
            run_str, fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "run" to be set to either '
            '"randomsearch" or "gridsearch"'
        )

        # Tests fixed_params type
        fixed_params_df = pd.DataFrame({})
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params_df, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "fixed_params" to be a dictionary '
            'of parameter values with which to run the selected classifier '
            'algorithm'
        )

        # Test tuned_params type
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        selected_features_list = ['X', 'X']
        splits_gen = create_generator(x_train_array.shape[0])
        n_components_pca_int = x_train_array.shape[1]
        run_str = 'train'
        fixed_params_dict = {'dual': False}
        tuned_params_list = []
        with self.assertRaises(TypeError) as message: check_arguments(
            'run_ml', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_gen,
            resampling_method, n_components_pca_int, run_str, fixed_params_dict,
            tuned_params_list, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Expect "tuned_params" to be a dictionary '
            'of parameter names (keys) and ranges of values to optimise '
            '(values) using either random or grid search'
        )

        # Test train_scoring_metric is string in list of recognised scoring
        # metrics in sklearn
        train_scoring_metric_str = 'mutual_info_score'  # Scoring metric used
        # for clustering, not classification
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric_str,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), '"train_scoring_metric" not recogised - '
            'please specify a string corresponding to the name of the metric '
            'you would like to use in the sklearn.metrics module, e.g. '
            '"accuracy".\nExpect metric to be in the following list:\n'
            '[\'accuracy\', \'balanced_accuracy\', \'top_k_accuracy\', '
            '\'average_precision\', \'neg_brier_score\', \'f1\', \'f1_micro\', '
            '\'f1_macro\', \'f1_weighted\', \'f1_samples\', \'neg_log_loss\', '
            '\'precision\', \'precision_micro\', \'precision_macro\', '
            '\'precision_weighted\', \'precision_samples\', \'recall\', '
            '\'recall_micro\', \'recall_macro\', \'recall_weighted\', '
            '\'recall_samples\', \'jaccard\', \'jaccard_micro\', '
            '\'jaccard_macro\', \'jaccard_weighted\', \'jaccard_samples\', '
            '\'roc_auc\', \'roc_auc_ovr\', \'roc_auc_ovo\', '
            '\'roc_auc_ovr_weighted\', \'roc_auc_ovo_weighted\']'
        )

        # Test test_scoring_funcs is a dictionary of scoring functions (keys)
        # and dictionaries of parameter values to run these functions with
        from sklearn.metrics import accuracy_score, jaccard_score
        test_scoring_funcs_dict = {accuracy_score: {'normalize': True},
                                   jaccard_score: {'average': 'weighted'}}
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs_dict, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Scoring function jaccard_score not '
            'recognised.\nExpect scoring functions to be in the following '
            'list:\n[\'accuracy_score\', \'f1_score\', \'precision_score\', '
            '\'recall_score\', \'roc_auc_score\', \'cohen_kappa_score\']'
        )

        # Test n_iter type is an integer
        n_iter_float = 3.0
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter_float, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), '"n_iter" should be set to a positive '
            'integer value'
        )

        # Test n_iter is a positive integer
        n_iter_int = -2
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter_int, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), '"n_iter" should be set to a positive '
            'integer value'
        )

        # Test cv_folds_inner_loop type is an integer
        cv_folds_inner_loop_dict = OrderedDict()
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop_dict,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "cv_folds_inner_loop" to be a '
            'positive integer value in the range of 2 - 10'
        )

        # Test cv_folds_inner_loop is an integer in the range of 2 - 10
        cv_folds_inner_loop_int = 11
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop_int,
            cv_folds_outer_loop, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "cv_folds_inner_loop" to be a '
            'positive integer value in the range of 2 - 10'
        )

        # Test cv_folds_outer_loop type is set to 'loocv' or an integer value
        cv_folds_outer_loop_float = 2.3
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop_float, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "cv_folds_outer_loop" to be set to '
            'either "loocv" (leave-one-out cross-validation) or a positive '
            'integer in the range of 2 - 10'
        )

        # Test cv_folds_outer_loop type is set to 'loocv' or an integer value
        cv_folds_outer_loop_str = ''
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop_str, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "cv_folds_outer_loop" to be set to '
            'either "loocv" (leave-one-out cross-validation) or a positive '
            'integer in the range of 2 - 10'
        )

        # Test cv_folds_outer_loop is an integer in the range of 2 - 10
        cv_folds_outer_loop_int = 1
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop_int, draw_conf_mat, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "cv_folds_outer_loop" to be set to '
            'either "loocv" (leave-one-out cross-validation) or a positive '
            'integer in the range of 2 - 10'
        )

        # Test draw_conf_mat type is a Boolean
        draw_conf_mat_float = 0.0
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat_float, plt_name, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "draw_conf_mat" to be a Boolean '
            'value (True or False)'
        )

        # Test plt_name
        plt_name_bool = False
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric,
            test_scoring_funcs, n_iter, cv_folds_inner_loop,
            cv_folds_outer_loop, draw_conf_mat, plt_name_bool, True
        )
        self.assertEqual(
            str(message.exception), 'Expect "plt_name" to be a string'
        )

        # Test passes with more complex default values
        from sklearn.metrics import precision_score
        x_train_ext = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        y_train_ext = np.array([2, 2, 2, 2, 2])
        train_groups_ext = np.array([3, 3, 3, 3, 3])
        x_test_ext = np.array([[4, 4, 4, 4]])
        y_test_ext = np.array([5])
        selected_features_ext = ['A', 'C', 'B']
        splits_ext = create_generator(x_train_ext.shape[0])
        resampling_method_ext = 'smote'
        n_components_pca_ext = 4
        run_ext = 'train'
        fixed_params_ext = {'randomstate': 0}
        tuned_params_ext = {'n_estimators': np.linspace(5, 50, 10)}
        train_scoring_metric_ext = 'precision'
        test_scoring_funcs_ext = {precision_score: {'average': 'macro'}}
        n_iter_ext = 100
        cv_folds_inner_loop_ext = 10
        cv_folds_outer_loop_ext = 'loocv'
        draw_conf_mat_ext = False
        plt_name_ext = 'run_ml'

        output_str = check_arguments(
            'PlaceHolder', x_train_ext, y_train_ext, train_groups_ext,
            x_test_ext, y_test_ext, selected_features_ext, splits_ext,
            resampling_method_ext, n_components_pca_ext, run_ext,
            fixed_params_ext, tuned_params_ext, train_scoring_metric_ext,
            test_scoring_funcs_ext, n_iter_ext, cv_folds_inner_loop_ext,
            cv_folds_outer_loop_ext, draw_conf_mat_ext, plt_name_ext, True
        )
        self.assertEqual(output_str, 'All checks passed')


    def test_split_train_test_data_random(self):
        """
        Tests split_train_test_data_random in train.py
        """

        print('Testing split_train_test_data_random')

    def test_split_train_test_data_user_defined(self):
        """
        Tests split_train_test_data_user_defined in train.py
        """

        print('Testing split_train_test_data_user_defined')

    def test_calc_feature_correlations(self):
        """
        Tests calc_feature_correlations in train.py
        """

        print('Testing calc_feature_correlations')

    def test_calc_feature_importances_kbest(self):
        """
        Tests calc_feature_importances_kbest in train.py
        """

        print('Testing calc_feature_importances_kbest')

    def test_calc_feature_importances_tree(self):
        """
        Tests calc_feature_importances_tree in train.py
        """

        print('Testing calc_feature_importances_tree')

    def test_calc_feature_importances_permutation(self):
        """
        Tests calc_feature_importances_permutation in train.py
        """

        print('Testing calc_feature_importances_permutation')

    def test_run_pca(self):
        """
        Tests run_pca in train.py
        """

        print('Testing run_pca')

    def test_plot_scatter_on_pca_axes(self):
        """
        Tests plot_scatter_on_pca_axes in train.py
        """

        print('Testing plot_scatter_on_pca_axes')

    def test_define_fixed_model_params(self):
        """
        Tests define_fixed_model_params in train.py
        """

        print('Testing define_fixed_model_params')

    def test_define_tuned_model_params(self):
        """
        Tests define_tuned_model_params in train.py
        """

        print('Testing define_tuned_model_params')

    def test_flag_extreme_params(self):
        """
        Tests flag_extreme_params in train.py
        """

        print('Testing flag_extreme_params')

    def test_conv_resampling_method(self):
        """
        Tests conv_resampling_method in train.py
        """

        print('Testing conv_resampling_method')

    def test_run_randomised_search(self):
        """
        Tests run_randomised_search in train.py
        """

        print('Testing run_randomised_search')

    def test_run_grid_search(self):
        """
        Tests run_grid_search in train.py
        """

        print('Testing run_grid_search')

    def test_train_model(self):
        """
        Tests train_model in train.py
        """

        print('Testing train_model')

    def test_test_model(self):
        """
        Tests test_model in train.py
        """

        print('Testing test_model')

    def test_run_ml(self):
        """
        Tests run_ml in train.py
        """

        print('Testing run_ml')

    def test_run_nested_CV(self):
        """
        Tests run_nested_CV in train.py
        """

        print('Testing run_nested_CV')

    def test_run_5x2_CV_paired_t_test(self):
        """
        Tests run_5x2_CV_paired_t_test in train.py
        """

        print('Testing run_5x2_CV_paired_t_test')
