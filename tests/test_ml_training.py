
# python -m unittest tests/test_ml_training.py

import copy
import numpy as np
import pandas as pd
import os
import shutil
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

        exp_input_dict = {
            1: [['A', 'B', 'C', 'D', 'B', 'A', 'D', 'C', 'C', 'A', 'D', 'B'],
                np.array([['A', 'C'], ['B', 'D']], dtype=object)],
            2: [np.array([['A', 'B', 'C', 'D'], ['B', 'A', 'D', 'C'], ['C', 'A', 'D', 'B']], dtype=object),
                np.array([['A', 'C'], ['B', 'D']], dtype=object)],
            3: [np.array(['A', 'B', 'C', 'D', 'B', 'A', 'D', 'C', 'C', np.nan, 'D', 'B'], dtype=object),
                np.array([['A', 'C'], ['B', 'D']], dtype=object)],
            4: [np.array(['A', 'B', 'C', 'D', 'B', 'A', 'D', 'C', 'C', 'A', 'D', 'B'], dtype=object),
                [['A', 'C'], ['B', 'D']]],
            5: [np.array(['A', 'B', 'C', 'D', 'B', 'A', 'D', 'C', 'C', 'A', 'D', 'B'], dtype=object),
                np.array([[np.nan, 'C'], ['B', 'D']], dtype=object)],
            6: [np.array(['A', 'B', 'C', 'D', 'B', 'A', 'D', 'C', 'C', 'A', 'D', 'B'], dtype=object),
                np.array([['A', 'C'], ['B', 'A']], dtype=object)],
            7: [np.array(['A', 'B', 'C', 'D', 'B', 'A', 'D', 'E', 'C', 'A', 'D', 'B'], dtype=object),
                np.array([['A', 'C'], ['B', 'D']], dtype=object)],
            8: [np.array(['A', 'B', 'C', 'D', 'B', 'A', 'D', 'E', 'C', 'A', 'D', 'B'], dtype=object),
                np.array([['A', 'C'], ['B', 'D'], ['E', 'F']], dtype=object)],
            9: [np.array(['A', 'B', 'C', 'D', 'B', 'A', 'D', 'C', 'C', 'A', 'D', 'B'], dtype=object),
                np.array([['A', 'C'], ['B', 'D']], dtype=object)]
        }

        for num in exp_input_dict.keys():
            subclasses = exp_input_dict[num][0]
            subclass_splits = exp_input_dict[num][1]

            if num == 1:
                with self.assertRaises(TypeError) as message:
                    splits = make_separate_subclass_splits(subclasses, subclass_splits)
                    next(splits)
                self.assertEqual(
                    str(message.exception), 'Expect "subclasses" to be a (1D) '
                    'array of subclass values'
                )
            elif num == 2:
                with self.assertRaises(ValueError) as message:
                    splits = make_separate_subclass_splits(subclasses, subclass_splits)
                    next(splits)
                self.assertEqual(
                    str(message.exception), 'Expect "subclasses" to be a 1D array'
                )
            elif num == 3:
                with self.assertRaises(ValueError) as message:
                    splits = make_separate_subclass_splits(subclasses, subclass_splits)
                    next(splits)
                self.assertEqual(
                    str(message.exception), 'NaN value(s) detected in '
                    '"subclasses" array'
                )
            elif num == 4:
                with self.assertRaises(TypeError) as message:
                    splits = make_separate_subclass_splits(subclasses, subclass_splits)
                    next(splits)
                self.assertEqual(
                    str(message.exception), 'Expect "subclass_splits" to be a '
                    '(2D) array of subclass values'
                )
            elif num == 5:
                with self.assertRaises(ValueError) as message:
                    splits = make_separate_subclass_splits(subclasses, subclass_splits)
                    next(splits)
                self.assertEqual(
                    str(message.exception), 'NaN value(s) detected in '
                    '"subclass_splits" array'
                )
            elif num == 6:
                with self.assertRaises(ValueError) as message:
                    splits = make_separate_subclass_splits(subclasses, subclass_splits)
                    next(splits)
                self.assertEqual(
                    str(message.exception), 'Repeated subclass labels detected '
                    'in "subclass_splits"'
                )
            elif num == 7:
                with self.assertRaises(ValueError) as message:
                    splits = make_separate_subclass_splits(subclasses, subclass_splits)
                    next(splits)
                self.assertEqual(
                    str(message.exception), 'Subclass E is found in '
                    '"subclasses" but not "subclass_splits"'
                )
            elif num == 8:
                with self.assertRaises(ValueError) as message:
                    splits = make_separate_subclass_splits(subclasses, subclass_splits)
                    next(splits)
                self.assertEqual(
                    str(message.exception), 'Subclass F is found in '
                    '"subclass_splits" but not "subclasses"'
                )
            elif num == 9:
                exp_split = (sub_list for sub_list in
                             [np.array([0, 2, 5, 7, 8, 9]),
                              np.array([1, 3, 4, 6, 10, 11])])
                act_split = make_separate_subclass_splits(subclasses, subclass_splits)
                for i, split_1 in enumerate(list(exp_split)):
                    for j, split_2 in enumerate(list(act_split)):
                        if i == j:
                            np.testing.assert_equal(split_1, split_2)

    def test_bootstrap_data(self):
        """
        Tests bootstrap_data in train.py
        """

        print('Testing bootstrap_data')

        exp_input_dict = {
            1: [[[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                 [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                 [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                 [1.6, 0.5, 1.0]],
                np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
                ['1', '2', '3'], True],
            2: [np.array([[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                          [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                          [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                          [1.6, 0.5, 1.0]]),
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                ['1', '2', '3'], True],
            3: [np.array([[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                          [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                          [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                          [1.6, 0.5, 1.0]]),
                np.array([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']]),
                ['1', '2', '3'], True],
            4: [np.array([[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                          [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                          [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                          [1.6, 0.5, 1.0]]),
                np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']),
                ['1', '2', '3'], True],
            5: [np.array([[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                          [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                          [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                          [1.6, 0.5, 1.0]]),
                np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
                np.array(['1', '2', '3']), True],
            6: [np.array([[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                          [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                          [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                          [1.6, 0.5, 1.0]]),
                np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
                ['1', '2', '3', '4'], True],
            7: [np.array([[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                          [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                          [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                          [1.6, 0.5, 1.0]]),
                np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
                ['1', '2', '3'], 1.0],
            8: [np.array([[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                          [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                          [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                          [1.6, 0.5, 1.0]]),
                np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
                ['1', '2', '3'], False],
            9: [np.array([[1.0, 1.5, 1.2], [4.6, 2.3, 2.1], [1.0, 2.2, 1.8],
                          [1.8, 1.1, 0.6], [0.7, 0.9, 0.7], [4.1, 3.3, 2.6],
                          [3.4, 2.5, 1.4], [2.7, 2.2, 1.9], [4.0, 4.0, 3.1],
                          [1.6, 0.5, 1.0]]),
                np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
                ['1', '2', '3'], True]
        }

        for num in exp_input_dict.keys():
            x = exp_input_dict[num][0]
            y = exp_input_dict[num][1]
            features = exp_input_dict[num][2]
            scale = exp_input_dict[num][3]

            if num == 1:
                with self.assertRaises(TypeError) as message: bootstrap_data(
                    x, y, features, scale, True
                )
                self.assertEqual(
                    str(message.exception), 'Expect "x" to be a (2D) array of x'
                    ' values'
                )
            if num == 2:
                with self.assertRaises(TypeError) as message: bootstrap_data(
                    x, y, features, scale, True
                )
                self.assertEqual(
                    str(message.exception), 'Expect "y" to be a (1D) array of y'
                    ' values'
                )
            if num == 3:
                with self.assertRaises(ValueError) as message: bootstrap_data(
                    x, y, features, scale, True
                )
                self.assertEqual(
                    str(message.exception), 'Expect "y" to be a 1D array of y '
                    'values'
                )
            if num == 4:
                with self.assertRaises(ValueError) as message: bootstrap_data(
                    x, y, features, scale, True
                )
                self.assertEqual(
                    str(message.exception), 'Different numbers of rows in '
                    'arrays "x" and "y"'
                )
            if num == 5:
                with self.assertRaises(TypeError) as message: bootstrap_data(
                    x, y, features, scale, True
                )
                self.assertEqual(
                    str(message.exception), 'Expect "features" to be a list'
                )
            if num == 6:
                with self.assertRaises(ValueError) as message: bootstrap_data(
                    x, y, features, scale, True
                )
                self.assertEqual(
                    str(message.exception), 'Expect entries in "features" list '
                    'to correspond to the columns in "x"'
                )
            if num == 7:
                with self.assertRaises(TypeError) as message: bootstrap_data(
                    x, y, features, scale, True
                )
                self.assertEqual(
                    str(message.exception), 'Expect "scale" to be a Boolean '
                    'value (either True or False)'
                )
            if num == 8:
                exp_out_x = pd.DataFrame(
                    np.array([[1.0, 1.5, 1.2],
                              [3.4, 2.5, 1.4],
                              [4.6, 2.3, 2.1],
                              [1.8, 1.1, 0.6],
                              [0.7, 0.9, 0.7],
                              [4.1, 3.3, 2.6],
                              [4.0, 4.0, 3.1],
                              [1.0, 1.5, 1.2],
                              [3.4, 2.5, 1.4],
                              [4.1, 3.3, 2.6]]),
                    index=None, columns=features
                )
                exp_out_y = ['a', 'g', 'b', 'd', 'e', 'f', 'i', 'a', 'g', 'f']
                act_out_x, act_out_y = bootstrap_data(x, y, features, scale, True)
                pd.testing.assert_frame_equal(exp_out_x, act_out_x)
                self.assertEqual(exp_out_y, act_out_y)
            if num == 9:
                exp_out_x = pd.DataFrame(
                    np.array([[-0.83478261, -0.5625, -0.15686275],
                              [0., 0.0625, 0.],
                              [0.4173913, -0.0625, 0.54901961],
                              [-0.55652174, -0.8125, -0.62745098],
                              [-0.93913043, -0.9375, -0.54901961],
                              [0.24347826, 0.5625, 0.94117647],
                              [0.20869565, 1., 1.33333333],
                              [-0.83478261, -0.5625, -0.15686275],
                              [0., 0.0625, 0.],
                              [0.24347826, 0.5625, 0.94117647]]),
                    index=None, columns=features
                )
                exp_out_y = ['a', 'g', 'b', 'd', 'e', 'f', 'i', 'a', 'g', 'f']
                act_out_x, act_out_y = bootstrap_data(x, y, features, scale, True)
                pd.testing.assert_frame_equal(exp_out_x, act_out_x)
                self.assertEqual(exp_out_y, act_out_y)

    def test_make_feat_importance_plots(self):
        """
        Tests make_feat_importance_plots in train.py
        """

        print('Testing make_feat_importance_plots')

        input_feat_importances = {
            'Feature_1': [7.8, 8.7, 0.1, 8.1, 0.4],
            'Feature_2': [6.4, 0.1, 0.6, 8.3, 5.2],
            'Feature_3': [7.1, 8.4, 0.0, 9.3, 2.5],
            'Feature_4': [3.4, 2.1, 1.6, 5.6, 9.4],
            'Feature_5': [8.5, 3.4, 6.6, 6.4, 9.0],
            'Feature_6': [3.5, 4.3, 8.9, 2.3, 4.1],
            'Feature_7': [6.5, 8.4, 2.1, 3.2, 7.8],
            'Feature_8': [8.2, 4.7, 4.3, 1.0, 4.3],
            'Feature_9': [8.2, 5.6, 5.0, 0.8, 0.9],
            'Feature_10': [1.9, 4.0, 0.5, 6.0, 7.8]
        }

        input_results_dir = 'tests/Temp_output'
        input_plt_name = 'PlaceHolder'

        for num in range(1, 7):
            if num == 1:
                with self.assertRaises(FileNotFoundError) as message:
                    make_feat_importance_plots(
                        input_feat_importances, input_results_dir,
                        input_plt_name, True
                    )
                self.assertEqual(
                    str(message.exception),
                    'Directory {} does not exist'.format(input_results_dir)
                )
            elif num == 2:
                os.mkdir(input_results_dir)
                with open('{}/{}_feat_importance_percentiles.svg'.format(
                    input_results_dir, input_plt_name
                ), 'w') as f:
                    f.write('PlaceHolder')
                with self.assertRaises(FileExistsError) as message:
                    make_feat_importance_plots(
                        input_feat_importances, input_results_dir,
                        input_plt_name, True
                    )
                self.assertEqual(
                    str(message.exception),
                    'File {}/{}_feat_importance_percentiles.svg already exists '
                    '- please rename this file so it is not overwritten by '
                    'running this function'.format(input_results_dir, input_plt_name)
                )
                shutil.rmtree(input_results_dir)
            elif num == 3:
                os.mkdir(input_results_dir)
                with open('{}/{}_feat_importance_all_data.svg'.format(
                    input_results_dir, input_plt_name
                ), 'w') as f:
                    f.write('PlaceHolder')
                with self.assertRaises(FileExistsError) as message:
                    make_feat_importance_plots(
                        input_feat_importances, input_results_dir,
                        input_plt_name, True
                    )
                self.assertEqual(
                    str(message.exception),
                    'File {}/{}_feat_importance_all_data.svg already exists - '
                    'please rename this file so it is not overwritten by '
                    'running this function'.format(input_results_dir, input_plt_name)
                )
                shutil.rmtree(input_results_dir)
            elif num == 4:
                os.mkdir(input_results_dir)
                with self.assertRaises(TypeError) as message:
                    make_feat_importance_plots(
                        pd.DataFrame({}), input_results_dir, input_plt_name, True
                    )
                self.assertEqual(
                    str(message.exception),
                    'Expect "feature_importances" to be a dictionary of '
                    'importance scores'
                )
                shutil.rmtree(input_results_dir)
            elif num == 5:
                os.mkdir(input_results_dir)
                with self.assertRaises(TypeError) as message:
                    make_feat_importance_plots(
                        input_feat_importances, input_results_dir, 1.0, True
                    )
                self.assertEqual(
                    str(message.exception),
                    'Expect "plt_name" to a string to append to the start of '
                    'the names of the saved plots'
                )
                shutil.rmtree(input_results_dir)
            elif num == 6:
                os.mkdir(input_results_dir)
                exp_importance_df = pd.DataFrame({
                    'Feature': ['Feature_1', 'Feature_3', 'Feature_5', 'Feature_7',
                                'Feature_2', 'Feature_9', 'Feature_8', 'Feature_6',
                                'Feature_10', 'Feature_4'],
                    'Score': [7.8, 7.1, 6.6, 6.5, 5.2, 5.0, 4.3, 4.1, 4.0, 3.4]
                })
                exp_cols = [
                    'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5',
                    'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10'
                ]
                exp_cols_all = [
                    'Feature_1', 'Feature_1', 'Feature_1', 'Feature_1', 'Feature_1',
                    'Feature_2', 'Feature_2', 'Feature_2', 'Feature_2', 'Feature_2',
                    'Feature_3', 'Feature_3', 'Feature_3', 'Feature_3', 'Feature_3',
                    'Feature_4', 'Feature_4', 'Feature_4', 'Feature_4', 'Feature_4',
                    'Feature_5', 'Feature_5', 'Feature_5', 'Feature_5', 'Feature_5',
                    'Feature_6', 'Feature_6', 'Feature_6', 'Feature_6', 'Feature_6',
                    'Feature_7', 'Feature_7', 'Feature_7', 'Feature_7', 'Feature_7',
                    'Feature_8', 'Feature_8', 'Feature_8', 'Feature_8', 'Feature_8',
                    'Feature_9', 'Feature_9', 'Feature_9', 'Feature_9', 'Feature_9',
                    'Feature_10', 'Feature_10', 'Feature_10', 'Feature_10', 'Feature_10'
                ]
                exp_all_vals = [
                    7.8, 8.7, 0.1, 8.1, 0.4, 6.4, 0.1, 0.6, 8.3, 5.2, 7.1, 8.4,
                    0.0, 9.3, 2.5, 3.4, 2.1, 1.6, 5.6, 9.4, 8.5, 3.4, 6.6, 6.4,
                    9.0, 3.5, 4.3, 8.9, 2.3, 4.1, 6.5, 8.4, 2.1, 3.2, 7.8, 8.2,
                    4.7, 4.3, 1.0, 4.3, 8.2, 5.6, 5.0, 0.8, 0.9, 1.9, 4.0, 0.5,
                    6.0, 7.8]
                exp_median_vals = [7.8, 5.2, 7.1, 3.4, 6.6, 4.1, 6.5, 4.3, 5.0, 4.0]
                exp_lower_conf_limit_vals = [
                    0.13, 0.15, 0.25, 1.65, 3.7, 2.42, 2.21, 1.33, 0.81, 0.64
                ]
                exp_upper_conf_limit_vals = [
                    8.64, 8.11, 9.21, 9.02, 8.95, 8.44, 8.34, 7.85, 7.94, 7.62
                ]
                (
                    act_importance_df, act_cols, act_cols_all, act_all_vals,
                    act_median_vals, act_lower_conf_limit_vals,
                    act_upper_conf_limit_vals
                ) = make_feat_importance_plots(
                    input_feat_importances, input_results_dir, input_plt_name,
                    True
                )
                pd.testing.assert_frame_equal(exp_importance_df, act_importance_df)
                self.assertEqual(exp_cols, act_cols)
                self.assertEqual(exp_cols_all, act_cols_all)
                np.testing.assert_almost_equal(exp_all_vals, act_all_vals, 7)
                np.testing.assert_almost_equal(
                    exp_median_vals, act_median_vals, 7
                )
                np.testing.assert_almost_equal(
                    exp_lower_conf_limit_vals, act_lower_conf_limit_vals, 7
                )
                np.testing.assert_almost_equal(
                    exp_upper_conf_limit_vals, act_upper_conf_limit_vals, 7
                )

                shutil.rmtree(input_results_dir)

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
        splits = [(y_train, np.array([]))]
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
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        selected_features_list = ['X', 'X']
        splits_gen = create_generator(x_train_array.shape[0])
        with self.assertRaises(TypeError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_gen,
            resampling_method, n_components_pca, run, fixed_params,
            tuned_params, train_scoring_metric, test_scoring_funcs, n_iter,
            cv_folds_inner_loop, cv_folds_outer_loop, draw_conf_mat, plt_name,
            True
        )
        self.assertEqual(
            str(message.exception), 'Expect "splits" to be a list of train/test'
            ' splits'
        )

        # Tests splits list matches dimensions of x_train
        x_train_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        y_train_array = np.array([2, 2, 2, 2])
        train_groups_array = np.array([3, 3, 3, 3])
        x_test_array = np.array([[4, 4], [4, 4]])
        y_test_array = np.array([5, 5])
        selected_features_list = ['X', 'X']
        splits_list = [(np.array([6, 6, 6]), np.array([6])),
                       (np.array([]), np.array([6, 6, 6, 6])),
                       (np.array([6]), np.array([6, 6, 6, 6]))]
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_list,
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
        splits_list = [(np.array([6, 6, 6]), np.array([6])),
                       (np.array([]), np.array([6, 6, 6, 6])),
                       (np.array([6]), np.array([6, 6, 6]))]
        n_components_pca_int = x_train_array.shape[1] + 1
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_list,
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
        splits_list = [(np.array([]), np.array([])),
                       (np.array([]), np.array([])),
                       (np.array([]), np.array([]))]
        n_components_pca_int = x_test_array.shape[1]
        run_str = 'random search'
        with self.assertRaises(ValueError) as message: check_arguments(
            'run_ml', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_list,
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
        splits_list = [(np.array([6, 6, 6]), np.array([6])),
                       (np.array([]), np.array([6, 6, 6, 6])),
                       (np.array([6]), np.array([6, 6, 6]))]
        n_components_pca_int = x_train_array.shape[1]
        run_str = 'train'
        fixed_params_dict = {'dual': False}
        tuned_params_list = []
        with self.assertRaises(TypeError) as message: check_arguments(
            'run_ml', x_train_array, y_train_array, train_groups_array,
            x_test_array, y_test_array, selected_features_list, splits_list,
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
        from sklearn.metrics import accuracy_score, jaccard_score, make_scorer
        train_scoring_metric_func = make_scorer(accuracy_score)
        test_scoring_funcs_dict = {accuracy_score: {'normalize': True},
                                   jaccard_score: {'average': 'weighted'}}
        with self.assertRaises(ValueError) as message: check_arguments(
            'PlaceHolder', x_train, y_train, train_groups, x_test, y_test,
            selected_features, splits, resampling_method, n_components_pca, run,
            fixed_params, tuned_params, train_scoring_metric_func,
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
        splits_ext = [(np.array([6, 6, 6, 6, 6]), np.array([]))]
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


    def test_class_initialisation(self):
        """
        Tests initialisation of RunML class
        """

        print('Testing RunML class')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({})
        classes = None
        subclasses = None
        shuffle = True

        # Test recognises that output directory already exists
        os.mkdir(results_dir)
        with self.assertRaises(FileExistsError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data, classes, subclasses, shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'Directory {} already found in {}'.format(results_dir, os.getcwd())
        )
        shutil.rmtree('tests/Temp_output')

        # Test "classes" must be None or a list
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(TypeError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data, np.array([]), subclasses, shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'Expect "classes" argument to be set either to None or to a list'
        )

        # Test "subclasses" must be None or a list
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(TypeError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data, [], np.array([]), shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'Expect "subclasses" argument to be set either to None or to a list'
        )

        # Test that if "subclasses" is set to a value other than None, classes
        # cannot be set to None
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(TypeError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data, classes, np.array([]), shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'If "subclasses" is set to a value other than None, then "classes" '
            'must also be set to a value other than None'
        )

        # Tests that if subclasses list is defined, the entries in the list
        # are formatted as "class_subclass"
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(ValueError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data, ['A', 'B', 'A', 'B'],
                ['A_1', 'B_1_', 'A_2', 'B_2'], shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'Entries in subclass list should be formatted as "class_subclass" '
            '(in which neither "class" nor "subclass" contains the character '
            '"_")'
        )

        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(ValueError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data, ['A', 'B', 'A', 'B'],
                ['A_1', 'C_1', 'A_2', 'B_2'], shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'Entries in subclass list should be formatted as "class_subclass" '
            '(in which neither "class" nor "subclass" contains the character '
            '"_")'
        )

        # Test requires "Analyte" column in fluor_data if classes is set to None
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(KeyError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data, classes, subclasses, shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            '\'No "Analyte" column detected in input dataframe - if you '
            'do not define the "classes" argument, the input dataframe'
            ' must contain an "Analyte" column\''
        )

        # Tests that number of entries in "classes" and "subclasses" lists are
        # equal to one another and to the number of rows in "fluor_data"
        # dataframe
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        fluor_data_df = pd.DataFrame({'Feature_1': [1, 3, 2, 4],
                                      'Feature_2': [2, 4, 3, 1]})
        classes_list = ['A', 'B', 'A', 'B']
        subclasses_list = ['A_1', 'B_1', 'A_2', 'B_2']
        with self.assertRaises(ValueError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data_df, ['A', 'B', 'A', 'B', 'A'],
                subclasses_list, shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'Mismatch between number of entries in the input dataframe and '
            'the "classes" list'
        )

        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(ValueError) as message:
            test_ml_train = RunML(
                results_dir, fluor_data_df, classes_list, ['A_1', 'B_1', 'B_2'],
                shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'Mismatch between number of entries in the input dataframe and '
            'the "subclasses" list'
        )

        # Tests prevents overwriting of "Classes" or "Subclasses" columns in
        # "fluor_data" dataframe
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(NameError) as message:
            test_ml_train = RunML(
                results_dir,
                pd.DataFrame({'Feature_1': [1, 3, 2, 4], 'Subclasses': [2, 4, 3, 1]}),
                classes_list, subclasses_list, shuffle, True
            )
        self.assertEqual(
            str(message.exception),
            'Please rename any columns in input dataframe labelled either '
            '"Classes" or "Subclasses", as these columns are added to the '
            'dataframe by the code during data processing'
        )

        # Tests no NaN or non-numeric values in "fluor_data" dataframe
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(ValueError) as message:
            test_ml_train = RunML(
                results_dir,
                pd.DataFrame({'Feature_1': [1, 3.0, 2, 4], 'Feature_2': [2, 4, np.nan, 1]}),
                ['A', 'B', 'A', 'B'], [np.nan, np.nan, np.nan, np.nan], shuffle, True
            )
        self.assertEqual(
            str(message.exception), 'NaN detected in input dataframe'
        )

        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(ValueError) as message:
            test_ml_train = RunML(
                results_dir,
                pd.DataFrame({'Feature_1': [1, 3.0, 2, '4'], 'Feature_2': [2, 4, 3, 1]}),
                ['A', 'B', 'A', 'B'], [np.nan, np.nan, np.nan, np.nan], shuffle, True
            )
        self.assertEqual(
            str(message.exception), 'Non-numeric value detected in input dataframe'
        )

        # Tests no NaN values in "classes" list
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(ValueError) as message:
            test_ml_train = RunML(
                results_dir,
                pd.DataFrame({'Feature_1': [1, 3.0, 2, 4], 'Feature_2': [2, 4, 3, 1]}),
                ['A', 'B', 'A', np.nan], [np.nan, np.nan, np.nan, np.nan], shuffle, True
            )
        self.assertEqual(
            str(message.exception), 'NaN detected in class values'
        )

        # Tests "subclasses" list is not a mixture of NaN and other values
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(ValueError) as message:
            test_ml_train = RunML(
                results_dir,
                pd.DataFrame({'Feature_1': [1, 3.0, 2, 4], 'Feature_2': [2, 4, 3, 1]}),
                ['A', 'B', 'A', 'B'], [np.nan, 1.0, np.nan, np.nan], shuffle, True
            )
        self.assertEqual(
            str(message.exception), 'NaN detected in subclass values'
        )

        # Tests that "shuffle" is a Boolean
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        with self.assertRaises(TypeError) as message:
            test_ml_train = RunML(
                results_dir,
                pd.DataFrame({'Feature_1': [1, 3.0, 2, 4], 'Feature_2': [2, 4, 3, 1]}),
                ['A', 'B', 'A', 'B'], [np.nan, np.nan, np.nan, np.nan], [], True
            )
        self.assertEqual(
            str(message.exception),
            'Expect "shuffle" to be a Boolean value (True or False)'
        )

        # Tests object attributes saved by RunML
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        test_ml_train = RunML(
            results_dir,
            pd.DataFrame({'Feature_1': [1.0, 3.0, 2.0, 4.0],
                          'Feature_2': [2.0, 4.0, 3.0, 1.0],
                          'Analyte': ['A', 'B', 'A', 'B']}),
            None, None, False, True
        )
        # Check self.classes is a numpy array of class labels (dtype=str)
        np.testing.assert_equal(np.array(['A', 'B', 'A', 'B']), test_ml_train.classes)
        # Check that self.sub_classes is either a numpy array of subclass labels
        # (dtype=str) or None
        self.assertEqual(None, test_ml_train.sub_classes)
        # Check that self.fluor_data is a dataframe
        pd.testing.assert_frame_equal(
            pd.DataFrame({'Feature_1': [1.0, 3.0, 2.0, 4.0],
                          'Feature_2': [2.0, 4.0, 3.0, 1.0]}),
            test_ml_train.fluor_data
        )
        # Check that self.x is a numpy array of self.fluor_data
        np.testing.assert_equal(np.array([[1.0, 2.0],
                                          [3.0, 4.0],
                                          [2.0, 3.0],
                                          [4.0, 1.0]]),
                                test_ml_train.x)
        # Check that self.y is the same as self.classes
        np.testing.assert_equal(test_ml_train.classes, test_ml_train.y)

        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)

    def test_split_train_test_data_random(self):
        """
        Tests split_train_test_data_random in train.py
        """

        print('Testing split_train_test_data_random')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 7, 2, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [2, 6, 4, 6]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Default function arguments
        x = np.array([[1, 2], [3, 4]])
        y = np.array(['a', 'b'])
        percent_test = 0.2

        # Test x is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.split_train_test_data_random(
                [[1, 2], [3, 4]], y, percent_test, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be an array of x values'
        )

        # Test y is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.split_train_test_data_random(
                x, ['a', 'b'], percent_test, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be an array of y values'
        )

        # Test that dimensions of x and y values match
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_random(
                np.array([[1, 2], [3, 4], [5, 6]]), y, percent_test, True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch in the dimensions of the input '
            '"x" and "y" values'
        )

        # Test x doesn't contain any NaN values
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_random(
                np.array([[1, np.nan], [3, 4]]), y, percent_test, True
            )
        self.assertEqual(
            str(message.exception), 'NaN value(s) detected in "x" data'
        )

        # Test y doesn't contain any NaN values
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_random(
                x, np.array([np.nan, 'b'], dtype=object), percent_test, True
            )
        self.assertEqual(
            str(message.exception), 'NaN value(s) detected in "y" data'
        )

        # Test percent_test is a float/integer value
        with self.assertRaises(TypeError) as message:
            test_ml_train.split_train_test_data_random(
                x, y, True, True
            )
        self.assertEqual(
            str(message.exception), '"percent_test" argument should be set to a'
            ' float in the range 0 - 0.5'
        )

        # Test percent_test is in the range 0 - 0.5
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_random(
                x, y, 0.52, True
            )
        self.assertEqual(
            str(message.exception), '"percent_test" argument should be set to a'
            ' float in the range 0 - 0.5'
        )

        # Test stratified k-fold split (random seed has been set to fixed value
        # so split will be consistent during the test (but not when running the
        # code outside of unit tests))
        exp_split = [np.array([0, 2, 3, 4]), np.array([1])]
        act_split = test_ml_train.split_train_test_data_random(
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            np.array(['a', 'a', 'a', 'a', 'a']), 0.2, True
        )
        np.testing.assert_equal(exp_split[0], act_split[0])
        np.testing.assert_equal(exp_split[1], act_split[1])

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_split_train_test_data_user_defined(self):
        """
        Tests split_train_test_data_user_defined in train.py
        """

        print('Testing split_train_test_data_user_defined')

        def_results_dir = 'tests/Temp_output'
        def_fluor_data = pd.DataFrame({'Feature_1': [4, 7, 2, 9],
                                       'Feature_2': [6, 6, 2, 6],
                                       'Feature_3': [2, 6, 4, 6]})
        def_classes = ['A', 'B', 'B', 'A']
        def_subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        def_shuffle = False

        test_ml_train = RunML(
            def_results_dir, def_fluor_data, def_classes, def_subclasses,
            def_shuffle, True
        )

        # Default function arguments
        subclasses = np.array([
            'Green_Diplomat', 'Black_PGTips', 'Green_Diplomat',
            'Black_Dragonfly', 'Black_Yorkshire', 'Green_Dragonfly',
            'Black_Dragonfly', 'Green_Clipper', 'Black_PGTips',
            'Green_Diplomat', 'Green_Dragonfly', 'Black_PGTips',
            'Green_Clipper', 'Green_Diplomat', 'Green_Diplomat',
            'Black_Yorkshire', 'Black_Yorkshire', 'Black_PGTips',
            'Black_Dragonfly', 'Black_Dragonfly', 'Green_Dragonfly',
            'Green_Clipper', 'Black_Dragonfly', 'Black_PGTips'
        ], dtype=object)
        test_subclasses = np.array(
            ['Green_Dragonfly', 'Black_Yorkshire'], dtype=object
        )

        # Tests "subclasses" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.split_train_test_data_user_defined(
                list(subclasses), test_subclasses
            )
        self.assertEqual(
            str(message.exception), 'Expect "subclasses" to be a (1D) array of '
            'subclass values'
        )

        # Tests "subclasses" is a 1D numpy array
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_user_defined(
                np.array([[subclass] for subclass in list(subclasses)]),
                test_subclasses
            )
        self.assertEqual(
            str(message.exception), 'Expect "subclasses" to be a 1D array'
        )

        # Tests no NaN values in "subclasses"
        nan_subclasses = copy.deepcopy(subclasses)
        nan_subclasses[15] = np.nan
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_user_defined(
                nan_subclasses, test_subclasses
            )
        self.assertEqual(
            str(message.exception), 'NaN value(s) detected in "subclasses" array'
        )

        # Tests "test_subclasses" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.split_train_test_data_user_defined(
                subclasses, list(test_subclasses)
            )
        self.assertEqual(
            str(message.exception), 'Expect "test_subclasses" argument to be a'
            ' (1D) array of the subclass values that should be separated out '
            'into the test set'
        )

        # Tests "test_subclasses" is a 1D numpy array
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_user_defined(
                subclasses,
                np.array([[test_subclass] for test_subclass in list(test_subclasses)])
            )
        self.assertEqual(
            str(message.exception), 'Expect "test_subclasses" to be a 1D array'
        )

        # Tests no NaN values in "test_subclasses"
        nan_test_subclasses = copy.deepcopy(test_subclasses)
        nan_test_subclasses[1] = np.nan
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_user_defined(
                subclasses, nan_test_subclasses
            )
        self.assertEqual(
            str(message.exception),
            'NaN value(s) detected in "test_subclasses" array'
        )

        # Tests that all entries in "test_subclasses" are also included in
        # "subclasses"
        with self.assertRaises(ValueError) as message:
            test_ml_train.split_train_test_data_user_defined(
                np.array([subclass for subclass in subclasses
                          if subclass != 'Black_Yorkshire']),
                test_subclasses
            )
        self.assertEqual(
            str(message.exception),
            'Not all of the entries in the "test_subclasses" array are found in'
            ' the "subclasses" array. Expect "test_subclasses" argument to be a'
            ' (1D) array of the subclass values that should be separated out '
            'into the test set'
        )

        # Tests generation of user-defined split
        exp_split = [
            np.array([0, 1, 2, 3, 6, 7, 8, 9, 11, 12,
                      13, 14, 17, 18, 19, 21, 22, 23]),
            np.array([4, 5, 10, 15, 16, 20])
        ]
        act_split = test_ml_train.split_train_test_data_user_defined(
            subclasses, test_subclasses
        )
        np.testing.assert_equal(exp_split[0], act_split[0])
        np.testing.assert_equal(exp_split[1], act_split[1])

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_calc_feature_correlations(self):
        """
        Tests calc_feature_correlations in train.py
        """

        print('Testing calc_feature_correlations')

        def_results_dir = 'tests/Temp_output'
        def_fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                       'Feature_2': [6, 6, 2, 6],
                                       'Feature_3': [8, 6, 4, 2]})
        def_classes = ['A', 'B', 'B', 'A']
        def_subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        def_shuffle = False

        test_ml_train = RunML(
            def_results_dir, def_fluor_data, def_classes, def_subclasses,
            def_shuffle, True
        )

        # Default function arguments
        fluor_data = None
        correlation_coeff = 'kendall'
        plt_name = ''
        abs_vals = False

        # Tests "fluor_data" is a dataframe
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_correlations(
                def_fluor_data.to_numpy(), correlation_coeff, plt_name,
                abs_vals, True
            )
        self.assertEqual(
            str(message.exception), '"fluor_data" should be a dataframe of '
            'fluorescence readings'
        )

        # Tests "fluor_data" contains only integer/float values
        test_fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                        'Feature_2': [6, 6, 2, 6],
                                        'Feature_3': [8, 'x', 4, 2]}, dtype=object)
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_correlations(
                test_fluor_data, correlation_coeff, plt_name, abs_vals, True
            )
        self.assertEqual(
            str(message.exception), 'Non-numeric value(s) in "fluor_data" - '
            'expect all values in "fluor_data" to be integers / floats'
        )

        # Tests "fluor_data" doesn't contain any NaN values
        test_fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, np.nan],
                                        'Feature_2': [6, 6, 2, 6],
                                        'Feature_3': [8, 6, 4, 2]}, dtype=object)
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_correlations(
                test_fluor_data, correlation_coeff, plt_name, abs_vals, True
            )
        self.assertEqual(
            str(message.exception), 'NaN value(s) found in "fluor_data"'
        )

        # Tests "correlation_coefficient" is set to "kendall", "spearman" or
        # "pearson"
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_correlations(
                fluor_data, 'kendal', plt_name, abs_vals, True
            )
        self.assertEqual(
            str(message.exception), 'Value specified for "correlation_coeff" '
            'not recognised - should be set to "kendall", "spearman" or '
            '"pearson"'
        )

        # Tests "plt_name" is a string to be appended to the beginning of the
        # name of the saved plot
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_correlations(
                fluor_data, correlation_coeff, 1.0, abs_vals, True
            )
        self.assertEqual(
            str(message.exception), '"plt_name" should be a string value'
        )

        # Tests "abs_vals" is a Boolean
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_correlations(
                fluor_data, correlation_coeff, plt_name, [], True
            )
        self.assertEqual(
            str(message.exception), '"abs_vals" should be a Boolean value'
        )

        # Tests Kendall's Tau correlation coefficient
        exp_corr_matrix = pd.DataFrame({
            'Feature_1': [1.0, -0.23570226, -0.66666667],
            'Feature_2': [-0.23570226, 1.0, 0.23570226],
            'Feature_3': [-0.66666667, 0.23570226, 1.0]
        })
        exp_corr_matrix.index = ['Feature_1', 'Feature_2', 'Feature_3']
        act_corr_matrix = test_ml_train.calc_feature_correlations(
            fluor_data, 'kendall', plt_name, False, True
        )
        pd.testing.assert_frame_equal(exp_corr_matrix, act_corr_matrix)

        # Tests Spearman's rank correlation coefficient (with absolute readings)
        exp_corr_matrix = pd.DataFrame({
            'Feature_1': [1.0, 0.25819889, 0.80000000],
            'Feature_2': [0.25819889, 1.0, 0.25819889],
            'Feature_3': [0.80000000, 0.25819889, 1.0]
        })
        exp_corr_matrix.index = ['Feature_1', 'Feature_2', 'Feature_3']
        act_corr_matrix = test_ml_train.calc_feature_correlations(
            fluor_data, 'spearman', plt_name, True, True
        )
        pd.testing.assert_frame_equal(exp_corr_matrix, act_corr_matrix)

        # Tests Pearson's correlation coefficient
        exp_corr_matrix = pd.DataFrame({
            'Feature_1': [1.0, -0.32163376, -0.8304548],
            'Feature_2': [-0.32163376, 1.0, 0.25819889],
            'Feature_3': [-0.8304548, 0.25819889, 1.0]
        })
        exp_corr_matrix.index = ['Feature_1', 'Feature_2', 'Feature_3']
        act_corr_matrix = test_ml_train.calc_feature_correlations(
            fluor_data, 'pearson', plt_name, False, True
        )
        pd.testing.assert_frame_equal(exp_corr_matrix, act_corr_matrix)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_calc_feature_importances_kbest(self):
        """
        Tests calc_feature_importances_kbest in train.py
        """

        print('Testing calc_feature_importances_kbest')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Defines function arguments
        x = None
        y = None
        features = None
        method_classif = 'f_classif'
        num_repeats = 1000
        scale = True
        plt_name = ''

        # Test "x" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_kbest(
                fluor_data, y, features, method_classif, num_repeats, scale,
                plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "x" is a 2D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_kbest(
                fluor_data.to_numpy().flatten(), y, features, method_classif,
                num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "y" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_kbest(
                fluor_data.to_numpy(), classes, features, method_classif,
                num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be a (1D) array of class '
            'labels'
        )

        # Test "y" is a 1D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_kbest(
                x, fluor_data.to_numpy(), features, method_classif, num_repeats,
                scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be a (1D) array of class '
            'labels'
        )

        # Test "features" is a list
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_kbest(
                fluor_data.to_numpy(), np.array(classes), fluor_data.columns,
                method_classif, num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "features" to be a list of the '
            'column ids in "x"'
        )

        # Test that dimensions of "x" and "y" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_kbest(
                x, np.array([1, 2, 3]), features, method_classif, num_repeats,
                scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch between the number of rows in "x"'
            ' and the number of entries in "y"'
        )

        # Test that dimensions of "x" and "features" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_kbest(
                x, y, fluor_data.columns.tolist()[:2], method_classif,
                num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch between the number of columns in '
            '"x" and the number of column ids in "features"'
        )

        # Test "method_classif" is a recognised value
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_kbest(
                x, y, features, 'fclassif', num_repeats, scale, plt_name,
                True
            )
        self.assertEqual(
            str(message.exception), '"method_classif" should be set to either '
            '"f_classif" or "mutual_info_classif"'
        )

        # Test "num_repeats" is an integer
        from sklearn.feature_selection import f_classif
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_kbest(
                x, y, features, f_classif, 1000., scale, plt_name,
                True
            )
        self.assertEqual(
            str(message.exception), '"num_repeats" should be set to a positive '
            'integer value'
        )

        # Test "num_repeats" is a positive integer
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_kbest(
                x, y, features, method_classif, -4, scale, plt_name,
                True
            )
        self.assertEqual(
            str(message.exception), '"num_repeats" should be set to a positive '
            'integer value'
        )

        # Test "scale" is a Boolean
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_kbest(
                x, y, features, method_classif, num_repeats, 1, plt_name,
                True
            )
        self.assertEqual(
            str(message.exception), '"scale" should be set to a Boolean value'
        )

        # Test "plt_name" is a string
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_kbest(
                x, y, features, method_classif, num_repeats, scale, True,
                True
            )
        self.assertEqual(
            str(message.exception), '"plt_name" should be a string value'
        )

        # Test KBest feature selection
        exp_importance_df = pd.DataFrame({'Feature': ['Feature_1', 'Feature_2'],
                                          'Score': [0.91984923, 0.08015077]})
        act_importance_df = test_ml_train.calc_feature_importances_kbest(
            np.array([[2, 2],
                      [1, 10],
                      [2, 3],
                      [1, 1],
                      [2, 7],
                      [10, 6],
                      [7, 2],
                      [2, 2],
                      [1, 10],
                      [10, 6],
                      [2, 7],
                      [3, 8]]),
            np.array(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']),
            ['Feature_1', 'Feature_2'], 'f_classif', 10, True, '', True
        )
        pd.testing.assert_frame_equal(exp_importance_df, act_importance_df)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_calc_feature_importances_tree(self):
        """
        Tests calc_feature_importances_tree in train.py
        """

        print('Testing calc_feature_importances_tree')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Defines function arguments
        x = None
        y = None
        features = None
        num_repeats = 1000
        scale = True
        plt_name = ''

        # Test "x" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_tree(
                fluor_data, y, features, num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "x" is a 2D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_tree(
                fluor_data.to_numpy().flatten(), y, features, num_repeats,
                scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "y" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_tree(
                fluor_data.to_numpy(), classes, features, num_repeats, scale,
                plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be a (1D) array of class '
            'labels'
        )

        # Test "y" is a 1D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_tree(
                x, fluor_data.to_numpy(), features, num_repeats, scale,
                plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be a (1D) array of class '
            'labels'
        )

        # Test "features" is a list
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_tree(
                fluor_data.to_numpy(), np.array(classes), fluor_data.columns,
                num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "features" to be a list of the '
            'column ids in "x"'
        )

        # Test that dimensions of "x" and "y" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_tree(
                x, np.array([1, 2, 3]), features, num_repeats, scale, plt_name,
                True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch between the number of rows in "x"'
            ' and the number of entries in "y"'
        )

        # Test that dimensions of "x" and "features" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_tree(
                x, y, fluor_data.columns.tolist()[:2], num_repeats, scale,
                plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch between the number of columns in '
            '"x" and the number of column ids in "features"'
        )

        # Test "num_repeats" is an integer
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_tree(
                x, y, features, 1000., scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), '"num_repeats" should be set to a positive '
            'integer value'
        )

        # Test "num_repeats" is a positive integer
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_tree(
                x, y, features, -4, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), '"num_repeats" should be set to a positive '
            'integer value'
        )

        # Test "scale" is a Boolean
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_tree(
                x, y, features, num_repeats, 1, plt_name, True
            )
        self.assertEqual(
            str(message.exception), '"scale" should be set to a Boolean value'
        )

        # Test "plt_name" is a string
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_tree(
                x, y, features, num_repeats, scale, True, True
            )
        self.assertEqual(
            str(message.exception), '"plt_name" should be a string value'
        )

        # Test tree feature selection
        exp_importance_df = pd.DataFrame({'Feature': ['Feature_1', 'Feature_2'],
                                          'Score': [0.554, 0.446]})
        act_importance_df = test_ml_train.calc_feature_importances_tree(
            np.array([[2, 2],
                      [1, 10],
                      [2, 3],
                      [1, 1],
                      [2, 7],
                      [10, 6],
                      [7, 2],
                      [2, 2],
                      [1, 10],
                      [10, 6],
                      [2, 7],
                      [3, 8]]),
            np.array(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']),
            ['Feature_1', 'Feature_2'], 1000, False, '', True
        )
        pd.testing.assert_frame_equal(exp_importance_df, act_importance_df, atol=0.01)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_calc_feature_importances_permutation(self):
        """
        Tests calc_feature_importances_permutation in train.py
        """

        print('Testing calc_feature_importances_permutation')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Defines function arguments
        from sklearn.ensemble import AdaBoostClassifier
        x = None
        y = None
        features = None
        classifier = AdaBoostClassifier
        parameters = {'n_estimators': [10, 30, 100, 300, 1000]}
        model_metric = 'accuracy'
        num_repeats = 1000
        scale = True
        plt_name = ''

        # Test "x" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_permutation(
                fluor_data, y, features, classifier, parameters, model_metric,
                num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "x" is a 2D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_permutation(
                fluor_data.to_numpy().flatten(), y, features, classifier,
                parameters, model_metric, num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "y" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_permutation(
                fluor_data.to_numpy(), classes, features, classifier,
                parameters, model_metric, num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be a (1D) array of class '
            'labels'
        )

        # Test "y" is a 1D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, fluor_data.to_numpy(), features, classifier, parameters,
                model_metric, num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be a (1D) array of class '
            'labels'
        )

        # Test "features" is a list
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_permutation(
                fluor_data.to_numpy(), np.array(classes), fluor_data.columns,
                classifier, parameters, model_metric, num_repeats, scale,
                plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "features" to be a list of the '
            'column ids in "x"'
        )

        # Test that dimensions of "x" and "y" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, np.array([1, 2, 3]), features, classifier, parameters,
                model_metric, num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch between the number of rows in "x"'
            ' and the number of entries in "y"'
        )

        # Test that dimensions of "x" and "features" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, y, fluor_data.columns.tolist()[:2], classifier, parameters,
                model_metric, num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch between the number of columns in '
            '"x" and the number of column ids in "features"'
        )

        # Test "parameters" is a dictionary
        from sklearn.ensemble import RandomForestClassifier
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, y, fluor_data.columns.tolist(), RandomForestClassifier, [],
                model_metric, num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "parameters" to be a dictionary of '
            'parameter names (keys) and arrays of values to consider for them '
            '(values) in a grid search'
        )

        # Test "model_metrics" is a recognised string
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, y, fluor_data.columns.tolist(), RandomForestClassifier,
                OrderedDict(), 'acuracy', num_repeats, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Value provided for "model_metric" not '
            'recognised - please specify one of the strings in the list below:'
            '\n[\'accuracy\', \'balanced_accuracy\', \'top_k_accuracy\', '
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

        # Test "num_repeats" is an integer
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, y, fluor_data.columns.tolist(), classifier, OrderedDict(),
                'precision', 1000., scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), '"num_repeats" should be set to a positive '
            'integer value'
        )

        # Test "num_repeats" is a positive integer
        with self.assertRaises(ValueError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, y, fluor_data.columns.tolist(), classifier, OrderedDict(),
                model_metric, 0, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), '"num_repeats" should be set to a positive '
            'integer value'
        )

        # Test "scale" is a Boolean
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, y, fluor_data.columns.tolist(), classifier, OrderedDict(),
                model_metric, num_repeats, 1, plt_name, True
            )
        self.assertEqual(
            str(message.exception), '"scale" should be set to a Boolean value'
        )

        # Test "plt_name" is a string
        with self.assertRaises(TypeError) as message:
            test_ml_train.calc_feature_importances_permutation(
                x, y, fluor_data.columns.tolist(), classifier, OrderedDict(),
                model_metric, num_repeats, False, {}, True
            )
        self.assertEqual(
            str(message.exception), '"plt_name" should be a string value'
        )

        # Test permutation feature selection
        from sklearn.svm import SVC
        svc_params = OrderedDict({
            'C': np.logspace(-5, 15, num=41, base=2),
            'gamma': np.logspace(-15, 3, num=37, base=2),
            'kernel': ['rbf']
        })
        exp_importance_df = pd.DataFrame({'Feature': ['Feature_1', 'Feature_2'],
                                          'Score': [-0.05714286, -0.06666667]})
        act_importance_df = test_ml_train.calc_feature_importances_permutation(
            np.array([[2, 2],
                      [1, 10],
                      [2, 3],
                      [1, 1],
                      [2, 7],
                      [10, 6],
                      [7, 2],
                      [2, 2],
                      [1, 10],
                      [10, 6],
                      [2, 7],
                      [3, 8]]),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            ['Feature_1', 'Feature_2'], SVC, svc_params, 'f1', 10, True,
            '', True
        )
        pd.testing.assert_frame_equal(exp_importance_df, act_importance_df)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_run_pca(self):
        """
        Tests run_pca in train.py
        """

        print('Testing run_pca')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Defines function arguments
        x = None
        scale = True
        plt_name = ''

        # Test "x" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.run_pca(fluor_data, scale, plt_name, True)
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "x" is a 2D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.run_pca(
                fluor_data.to_numpy().flatten(), scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "scale" is a Boolean
        with self.assertRaises(TypeError) as message:
            test_ml_train.run_pca(x, 1, plt_name, True)
        self.assertEqual(
            str(message.exception), '"scale" should be set to a Boolean value'
        )

        # Test "plt_name" is a string
        with self.assertRaises(TypeError) as message:
            test_ml_train.run_pca(x, False, [], True)
        self.assertEqual(
            str(message.exception), '"plt_name" should be a string value'
        )

        # Test PCA calculation
        act_model = test_ml_train.run_pca(fluor_data.to_numpy(), True, '', True)

        # These values are for fitting the PCA to the unscaled data (because
        # transformation of the data is not carried out in run_pca)
        exp_transform_x = np.array(
            [[-3.18109341, -0.22506071,  1.44221788],
             [-3.49530274, -0.42244343, -1.36176368],
             [ 2.37050586,  2.57343631, -0.09015284],
             [ 4.30589029, -1.92593218,  0.00969865]]
        )
        np.testing.assert_almost_equal(
            exp_transform_x, act_model.fit_transform(fluor_data.to_numpy()), 7
        )
        np.testing.assert_almost_equal(
            np.array([0.76223747, 0.17312951, 0.06463302]),
            act_model.explained_variance_ratio_, 7
        )

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_plot_scatter_on_pca_axes(self):
        """
        Tests plot_scatter_on_pca_axes in train.py
        """

        print('Testing plot_scatter_on_pca_axes')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_3']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Defines function arguments
        x = None
        y = None
        sub_list = None
        num_dimensions = 2
        scale = True
        plt_name = ''

        # Test "x" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                fluor_data, y, sub_list, num_dimensions, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "x" is a 2D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                fluor_data.to_numpy().flatten(), y, sub_list, num_dimensions,
                scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "x" to be a (2D) array of '
            'fluorescence readings'
        )

        # Test "y" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                fluor_data.to_numpy(), classes, sub_list, num_dimensions, scale,
                plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be a (1D) array of class '
            'labels'
        )

        # Test "y" is a 1D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, fluor_data.to_numpy(), sub_list, num_dimensions, scale,
                plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "y" to be a (1D) array of class '
            'labels'
        )

        # Test "subclasses" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, fluor_data.to_numpy().flatten()[:4], subclasses,
                num_dimensions, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "subclasses" to be set either to '
            'None, or to a (1D) array of the subclasses present in the dataset'
        )

        # Test "subclasses" is a 1D array
        with self.assertRaises(ValueError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, np.array(classes), fluor_data.to_numpy(), num_dimensions,
                scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "subclasses" to be set either to '
            'None, or to a (1D) array of the subclasses present in the dataset'
        )

        # Test that dimensions of "x" and "y" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, np.array([1, 2, 3]), np.array(subclasses), num_dimensions,
                scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch between the number of rows in "x"'
            ' and the number of entries in "y"'
        )

        # Test that dimensions of "x" and "subclasses" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, y, np.array([1, 2, 3]), num_dimensions, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Mismatch between the number of rows in "x"'
            ' and the number of entries in "subclasses"'
        )

        # Test "num_dimensions" is equal to 2 or 3
        with self.assertRaises(ValueError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, y, sub_list, 4, scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Expect "num_dimensions" to be set to 2 or 3'
        )

        # Test "scale" is a Boolean
        with self.assertRaises(TypeError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, y, sub_list, 3, '', plt_name, True
            )
        self.assertEqual(
            str(message.exception), '"scale" should be a Boolean (True or False)'
        )

        # Test "plt_name" is a string
        with self.assertRaises(TypeError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, y, sub_list, num_dimensions, scale, [], True
            )
        self.assertEqual(
            str(message.exception), '"plt_name" should be a string value'
        )

        # Test subclasses contain '_' character only once
        with self.assertRaises(ValueError) as message:
            test_ml_train.plot_scatter_on_pca_axes(
                x, y, np.array(['A_1', 'B_1_', 'B_2', 'A_2_2']), num_dimensions,
                scale, plt_name, True
            )
        self.assertEqual(
            str(message.exception), 'Character "_" found in subclass A_2_2 '
            'more/less than once'
        )

        # Test function with defined subclasses
        exp_cat_class_colours = {'A': 'b',
                                 'B': 'g'}
        exp_cat_markers = {'A_1': 'o',
                           'A_3': 'v',
                           'B_1': 'o',
                           'B_2': 'v'}
        exp_cat_colours = {'A_1': 'b',
                           'A_3': 'b',
                           'B_1': 'g',
                           'B_2': 'g'}
        exp_X_reduced = np.array(
            [[-1.18870441, -0.758145263],
             [-1.16720749, -0.579369555],
             [3.03636219, -0.179547060],
             [-0.680450290, 1.51706188]]
        )

        (
            act_cat_class_colours, act_cat_markers, act_cat_colours,
            act_X_reduced
        ) = test_ml_train.plot_scatter_on_pca_axes(
            x, y, sub_list, num_dimensions, scale, plt_name, True
        )
        self.assertEqual(exp_cat_class_colours, act_cat_class_colours)
        self.assertEqual(exp_cat_markers, act_cat_markers)
        self.assertEqual(exp_cat_colours, act_cat_colours)
        np.testing.assert_almost_equal(exp_X_reduced, act_X_reduced)

        # Test function with no subclasses
        shutil.rmtree('tests/Temp_output')
        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = None
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        exp_cat_class_colours = {'A': 'b',
                                 'B': 'g'}
        exp_cat_markers = {'A': 'o',
                           'B': 'o'}
        exp_cat_colours = {'A': 'b',
                           'B': 'g'}
        exp_X_reduced = np.array(
            [[-3.18109341, -0.22506071, 1.44221788],
             [-3.49530274, -0.42244343, -1.36176368],
             [2.37050586, 2.57343631, -0.09015284],
             [4.30589029, -1.92593218, 0.00969865]]
        )

        (
            act_cat_class_colours, act_cat_markers, act_cat_colours,
            act_X_reduced
        ) = test_ml_train.plot_scatter_on_pca_axes(
            x, y, sub_list, 3, False, plt_name, True
        )
        self.assertEqual(exp_cat_class_colours, act_cat_class_colours)
        self.assertEqual(exp_cat_markers, act_cat_markers)
        self.assertEqual(exp_cat_colours, act_cat_colours)
        np.testing.assert_almost_equal(exp_X_reduced, act_X_reduced)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_define_fixed_model_params(self):
        """
        Tests define_fixed_model_params in train.py
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import AdaBoostRegressor

        print('Testing define_fixed_model_params')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Test LogisticRegression
        exp_params = OrderedDict({'n_jobs': -1})
        act_params = test_ml_train.define_fixed_model_params(LogisticRegression())
        self.assertEqual(exp_params, act_params)

        # Test KNeighborsClassifier
        exp_params = OrderedDict({'metric': 'minkowski',
                                  'n_jobs': -1})
        act_params = test_ml_train.define_fixed_model_params(KNeighborsClassifier())
        self.assertEqual(exp_params, act_params)

        # Test LinearSVC
        exp_params = OrderedDict({'dual': False})
        act_params = test_ml_train.define_fixed_model_params(LinearSVC())
        self.assertEqual(exp_params, act_params)

        # Test SVC
        exp_params = OrderedDict()
        act_params = test_ml_train.define_fixed_model_params(SVC())
        self.assertEqual(exp_params, act_params)

        # Test AdaBoostClassifier
        exp_params = OrderedDict()
        act_params = test_ml_train.define_fixed_model_params(AdaBoostClassifier())
        self.assertEqual(exp_params, act_params)

        # Test Gaussian Naive Bayes
        exp_params = OrderedDict()
        act_params = test_ml_train.define_fixed_model_params(GaussianNB())
        self.assertEqual(exp_params, act_params)

        # Test unexpected classifier
        with self.assertRaises(TypeError) as message:
            test_ml_train.define_fixed_model_params(AdaBoostRegressor())
        self.assertEqual(
            str(message.exception),
            'Unrecognised value provided for "classifier". Expect "classifier" '
            'to be one of:\nsklearn.linear_model.LogisticRegression()\n'
            'sklearn.neighbors.KNeighborsClassifier()\n'
            'sklearn.svm.LinearSVC()\nsklearn.svm.SVC()\n'
            'sklearn.ensemble.AdaBoostClassifier()\n'
            'sklearn.naive_bayes.GaussianNB()'
        )

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_define_tuned_model_params(self):
        """
        Tests define_tuned_model_params in train.py
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import AdaBoostRegressor

        print('Testing define_tuned_model_params')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Defines function arguments
        x_train = fluor_data.to_numpy()
        n_folds = 5

        # Test "x_train" is a numpy array
        with self.assertRaises(TypeError) as message:
            test_ml_train.define_tuned_model_params(
                LogisticRegression(), fluor_data, n_folds
            )
        self.assertEqual(
            str(message.exception), '"x_train" should be a (2D) array of '
            'fluoresence readings'
        )

        # Test "n_folds" is an integer
        with self.assertRaises(TypeError) as message:
            test_ml_train.define_tuned_model_params(
                LogisticRegression(), x_train, 5.0
            )
        self.assertEqual(
            str(message.exception), '"n_folds" should be set to a positive '
            'integer value'
        )

        # Test n_folds is a positive integer
        with self.assertRaises(ValueError) as message:
            test_ml_train.define_tuned_model_params(
                LogisticRegression(), x_train, 0
            )
        self.assertEqual(
            str(message.exception), '"n_folds" should be set to a positive '
            'integer value'
        )

        # Test LogisticRegression
        exp_params = OrderedDict({
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'sag', 'saga', 'newton-cg', 'lbfgs'],
            'multi_class': ['ovr', 'multinomial'],
            'C': np.logspace(-3, 5, 17)
        })
        act_params = test_ml_train.define_tuned_model_params(
            LogisticRegression(), x_train, n_folds
        )
        self.assertEqual(list(exp_params.keys()), list(act_params.keys()))
        for exp_key, exp_val in exp_params.items():
            act_val = act_params[exp_key]
            if type(exp_val) == np.ndarray:
                np.testing.assert_equal(exp_val, act_val)
            else:
                self.assertEqual(exp_val, act_val)

        # Test KNeighborsClassifier
        with self.assertRaises(AlgorithmError) as message:
            test_ml_train.define_tuned_model_params(
                KNeighborsClassifier(), x_train, n_folds
            )
        self.assertEqual(
            str(message.exception), 'Too few data points in dataset to run k '
            'nearest neighbours'
        )

        exp_params = OrderedDict({
            'n_neighbors': np.array([2, 3]),
            'weights': ['uniform', 'distance'],
            'p': np.array([1, 2])
        })
        act_params = test_ml_train.define_tuned_model_params(
            KNeighborsClassifier(), x_train, 1
        )
        self.assertEqual(list(exp_params.keys()), list(act_params.keys()))
        for exp_key, exp_val in exp_params.items():
            act_val = act_params[exp_key]
            if type(exp_val) == np.ndarray:
                np.testing.assert_equal(exp_val, act_val)
            else:
                self.assertEqual(exp_val, act_val)

        # Test LinearSVC
        exp_params = OrderedDict({'C': np.logspace(-5, 15, num=41, base=2)})
        act_params = test_ml_train.define_tuned_model_params(
            LinearSVC(), x_train, n_folds
        )
        self.assertEqual(list(exp_params.keys()), list(act_params.keys()))
        for exp_key, exp_val in exp_params.items():
            act_val = act_params[exp_key]
            if type(exp_val) == np.ndarray:
                np.testing.assert_equal(exp_val, act_val)
            else:
                self.assertEqual(exp_val, act_val)

        # Test SVC
        exp_params = OrderedDict({
            'C': np.logspace(-5, 15, num=41, base=2),
            'gamma': np.logspace(-15, 3, num=37, base=2),
            'kernel': ['rbf']
        })
        act_params = test_ml_train.define_tuned_model_params(
            SVC(), x_train, n_folds
        )
        self.assertEqual(list(exp_params.keys()), list(act_params.keys()))
        for exp_key, exp_val in exp_params.items():
            act_val = act_params[exp_key]
            if type(exp_val) == np.ndarray:
                np.testing.assert_equal(exp_val, act_val)
            else:
                self.assertEqual(exp_val, act_val)

        # Test AdaBoostClassifier
        with self.assertRaises(AlgorithmError) as message:
            test_ml_train.define_tuned_model_params(
                AdaBoostClassifier(), x_train, n_folds
            )
        self.assertEqual(
            str(message.exception), 'Too few data points in dataset to use '
            'AdaBoost classifier'
        )

        exp_params = OrderedDict({
            'n_estimators': np.array([int(x) for x in np.logspace(1, 4, 7)])
        })
        act_params = test_ml_train.define_tuned_model_params(
            AdaBoostClassifier(), x_train, 1
        )
        self.assertEqual(list(exp_params.keys()), list(act_params.keys()))
        for exp_key, exp_val in exp_params.items():
            act_val = act_params[exp_key]
            if type(exp_val) == np.ndarray:
                np.testing.assert_equal(exp_val, act_val)
            else:
                self.assertEqual(exp_val, act_val)

        # Test Gaussian Naive Bayes
        exp_params = OrderedDict()
        act_params = test_ml_train.define_tuned_model_params(
            GaussianNB(), x_train, n_folds
        )
        self.assertEqual(exp_params, act_params)

        # Test unexpected classifier
        with self.assertRaises(TypeError) as message:
            act_params = test_ml_train.define_tuned_model_params(
                AdaBoostRegressor(), x_train, n_folds
            )
        self.assertEqual(
            str(message.exception),
            'Unrecognised value provided for "classifier". Expect "classifier" '
            'to be one of:\nsklearn.linear_model.LogisticRegression()\n'
            'sklearn.neighbors.KNeighborsClassifier()\n'
            'sklearn.svm.LinearSVC()\nsklearn.svm.SVC()\n'
            'sklearn.ensemble.AdaBoostClassifier()\n'
            'sklearn.naive_bayes.GaussianNB()'
        )

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_flag_extreme_params(self):
        """
        Tests flag_extreme_params in train.py
        """

        print('Testing flag_extreme_params')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Defines function arguments
        best_params = {'A': 1,
                       'B': 1.5,
                       'C': 1.0}
        poss_params = OrderedDict({'B': np.array([1.5, 2.5]),
                                   'A': np.array([0, 1, 2, 3]),
                                   'C': np.array([1.0, 1.5, 2.0])})

        # Test "best_params" is a dictionary
        with self.assertRaises(TypeError) as message:
            test_ml_train.flag_extreme_params([], poss_params, True)
        self.assertEqual(
            str(message.exception), 'Expect "best_params" to be a dictionary of'
            ' "optimal" parameter values returned after running an algorithm '
            'such as RandomizedSearchCV or GridSearchCV'
        )

        # Test "poss_params" is a dictionary
        with self.assertRaises(TypeError) as message:
            test_ml_train.flag_extreme_params(best_params, True, True)
        self.assertEqual(
            str(message.exception), 'Expect "poss_params" to be the dictionary '
            'of parameter ranges fed into the optimisation algorithm, such as '
            'that returned by define_model_params function'
        )

        # Test keys in "best_params" and "poss_params" match
        with self.assertRaises(ValueError) as message:
            test_ml_train.flag_extreme_params({'A': 1}, poss_params, True)
        self.assertEqual(
            str(message.exception), 'Mismatch in the keys in "best_params" and '
            '"poss_params"'
        )

        # Test warning message that should be printed when parameter value in
        # "best_params" lies at the extreme end of the range specified in
        # "poss_params"
        exp_warning = (
            '\x1b[31m WARNING: Optimal value selected for C is at the extreme '
            'of the range tested \033[0m \nRange tested: [1.0, 1.5, 2.0]\nValue '
            'selected: 1.0\n\n'
        )
        act_warning = test_ml_train.flag_extreme_params(best_params, poss_params, True)
        self.assertEqual(exp_warning, act_warning)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_conv_resampling_method(self):
        """
        Tests conv_resampling_method in train.py
        """

        print('Testing conv_resampling_method')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({'Feature_1': [4, 2, 7, 9],
                                   'Feature_2': [6, 6, 2, 6],
                                   'Feature_3': [8, 6, 4, 2]})
        classes = ['A', 'B', 'B', 'A']
        subclasses = ['A_1', 'B_1', 'B_2', 'A_2']
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Tests error
        with self.assertRaises(ValueError) as message:
            test_ml_train.conv_resampling_method('')
        self.assertEqual(
            str(message.exception), 'Resampling method  not recognised'
        )

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_run_randomised_search(self):
        """
        Tests run_randomised_search in train.py
        """

        print('Testing run_randomised_search')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({
            'Feature_1': [5, 9, 8, 1, 3, 5, 10, 6, 7, 1, 8, 9, 1, 10, 2, 2, 8,
                          7, 1, 3, 8, 4, 3, 4, 4, 6, 2, 10, 4, 5, 1, 7, 10, 3,
                          10, 6, 3, 8, 1, 4, 6, 1, 5, 2, 2, 1, 7, 1, 2, 4],
            'Feature_2': [9, 9, 7, 9, 6, 4, 7, 4, 2, 9, 7, 9, 7, 6, 4, 10, 8, 1,
                          5, 4, 3, 3, 4, 3, 1, 4, 9, 6, 7, 10, 4, 6, 9, 2, 7, 4,
                          3, 5, 7, 10, 1, 5, 3, 7, 2, 5, 10, 2, 2, 5],
            'Feature_3': [5, 4, 8, 10, 3, 2, 10, 5, 1, 10, 5, 5, 5, 10, 7, 1, 8,
                          8, 2, 1, 10, 9, 10, 6, 7, 4, 3, 3, 10, 10, 4, 7, 4, 6,
                          10, 7, 6, 9, 4, 9, 9, 4, 4, 5, 10, 2, 10, 1, 7, 10]
        })
        classes = [
            'A', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'B',
            'A', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'A',
            'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'A',
            'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'B'
        ]
        subclasses = [
            'A_1', 'B_2', 'B_1', 'A_2', 'B_1', 'B_2', 'A_2', 'A_1', 'B_1',
            'A_1', 'A_2', 'B_1', 'B_2', 'A_1', 'A_1', 'A_2', 'B_2', 'B_1',
            'A_1', 'B_2', 'A_2', 'B_2', 'A_2', 'A_2', 'A_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'A_1', 'A_2', 'A_1', 'B_2', 'A_1', 'B_1', 'A_2',
            'B_2', 'B_2', 'A_1', 'A_1', 'B_1', 'B_1', 'B_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'B_2', 'B_1', 'B_1'
        ]
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Test random search with PCA
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GroupKFold

        parameters = {'n_estimators': [3, 10, 30, 100, 300, 1000]}
        splits = list(GroupKFold(n_splits=2).split(
            X=test_ml_train.x, y=test_ml_train.y,
            groups=test_ml_train.sub_classes
        ))

        exp_search_results = {
            'params': [{'AdaBoostClassifier__n_estimators': 30},
                       {'AdaBoostClassifier__n_estimators': 10},
                       {'AdaBoostClassifier__n_estimators': 300},
                       {'AdaBoostClassifier__n_estimators': 3}],
            'split0_test_score': np.array([0.68, 0.68, 0.68, 0.64]),
            'split1_test_score': np.array([0.68, 0.68, 0.6, 0.64]),
            'mean_test_score': np.array([0.68, 0.68, 0.64, 0.64]),
            'std_test_score': np.array([0., 0., 0.04, 0.]),
            'rank_test_score': np.array([1, 1, 3, 3], dtype=np.int32)
        }
        exp_best_params = {'AdaBoostClassifier__n_estimators': 30}

        act_random_search = test_ml_train.run_randomised_search(
            test_ml_train.x, test_ml_train.y, test_ml_train.sub_classes,
            ['Feature_1', 'Feature_3'], AdaBoostClassifier(random_state=1),
            splits, 'smote', 1, parameters, 'accuracy', 4, True
        )
        act_search_results = act_random_search.cv_results_
        for key in [
            'mean_fit_time', 'std_fit_time', 'mean_score_time',
            'std_score_time', 'param_AdaBoostClassifier__n_estimators'
        ]:
            del(act_search_results[key])
        act_best_params = act_random_search.best_params_

        self.assertEqual(list(exp_search_results.keys()),
                         list(act_search_results.keys()))
        for key in exp_search_results.keys():
            if type(exp_search_results[key]) == np.ndarray:
                np.testing.assert_almost_equal(
                    exp_search_results[key], act_search_results[key], 7
                )
            else:
                self.assertEqual(
                    exp_search_results[key], act_search_results[key]
                )
        self.assertEqual(exp_best_params, act_best_params)

        # Test random search without PCA
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, make_scorer

        parameters = {'C': np.logspace(-3, 5, num=9, base=10)}
        splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=1).split(
            X=test_ml_train.x, y=test_ml_train.y, groups=test_ml_train.sub_classes
        ))
        scoring_metric = make_scorer(f1_score, average='binary', pos_label='A')

        exp_search_results = {
            'params': [{'LinearSVC__C': 0.001},
                       {'LinearSVC__C': 0.01},
                       {'LinearSVC__C': 0.1},
                       {'LinearSVC__C': 1.0},
                       {'LinearSVC__C': 10.0},
                       {'LinearSVC__C': 100.0},
                       {'LinearSVC__C': 1000.0},
                       {'LinearSVC__C': 10000.0},
                       {'LinearSVC__C': 100000.0}],
            'split0_test_score': np.array([
                0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667,
                0.66666667, 0.66666667, 0.66666667, 0.66666667
            ]),
            'split1_test_score': np.array([
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ]),
            'split2_test_score': np.array([
                0.33333333, 0.33333333, 0.57142857, 0.57142857, 0.57142857,
                0.57142857, 0.57142857, 0.57142857, 0.57142857
            ]),
            'split3_test_score': np.array([
                0.44444444, 0.44444444, 0.6, 0.54545455, 0.54545455, 0.54545455,
                0.54545455, 0.54545455, 0.54545455
            ]),
            'mean_test_score': np.array([
                0.48611111, 0.48611111, 0.58452381, 0.57088745, 0.57088745,
                0.57088745, 0.57088745, 0.57088745, 0.57088745
            ]),
            'std_test_score': np.array([
                0.12028131, 0.12028131, 0.05979699, 0.0609217, 0.0609217,
                0.0609217, 0.0609217, 0.0609217, 0.0609217
            ]),
            'rank_test_score': np.array([8, 8, 1, 2, 2, 2, 2, 2, 2], dtype=np.int32)
        }
        exp_best_params = {'LinearSVC__C': 0.1}

        act_random_search = test_ml_train.run_randomised_search(
            test_ml_train.x, test_ml_train.y, None, ['Feature_1', 'Feature_2'],
            LinearSVC(dual=False, random_state=1), splits, 'max_sampling', None,
            parameters, scoring_metric, None, True
        )
        act_search_results = act_random_search.cv_results_
        for key in [
            'mean_fit_time', 'std_fit_time', 'mean_score_time',
            'std_score_time', 'param_LinearSVC__C'
        ]:
            del(act_search_results[key])
        act_best_params = act_random_search.best_params_

        self.assertEqual(list(exp_search_results.keys()),
                         list(act_search_results.keys()))
        for key in exp_search_results.keys():
            if type(exp_search_results[key]) == np.ndarray:
                np.testing.assert_almost_equal(
                    exp_search_results[key], act_search_results[key], 7
                )
            else:
                self.assertEqual(
                    exp_search_results[key], act_search_results[key]
                )
        self.assertEqual(exp_best_params, act_best_params)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_run_grid_search(self):
        """
        Tests run_grid_search in train.py
        """

        print('Testing run_grid_search')

        results_dir = 'tests/Temp_output'
        fluor_data = pd.DataFrame({
            'Feature_1': [5, 9, 8, 1, 3, 5, 10, 6, 7, 1, 8, 9, 1, 10, 2, 2, 8,
                          7, 1, 3, 8, 4, 3, 4, 4, 6, 2, 10, 4, 5, 1, 7, 10, 3,
                          10, 6, 3, 8, 1, 4, 6, 1, 5, 2, 2, 1, 7, 1, 2, 4],
            'Feature_2': [9, 9, 7, 9, 6, 4, 7, 4, 2, 9, 7, 9, 7, 6, 4, 10, 8, 1,
                          5, 4, 3, 3, 4, 3, 1, 4, 9, 6, 7, 10, 4, 6, 9, 2, 7, 4,
                          3, 5, 7, 10, 1, 5, 3, 7, 2, 5, 10, 2, 2, 5],
            'Feature_3': [5, 4, 8, 10, 3, 2, 10, 5, 1, 10, 5, 5, 5, 10, 7, 1, 8,
                          8, 2, 1, 10, 9, 10, 6, 7, 4, 3, 3, 10, 10, 4, 7, 4, 6,
                          10, 7, 6, 9, 4, 9, 9, 4, 4, 5, 10, 2, 10, 1, 7, 10]
        })
        classes = [
            'A', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'B',
            'A', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'A',
            'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'A',
            'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'B'
        ]
        subclasses = [
            'A_1', 'B_2', 'B_1', 'A_2', 'B_1', 'B_2', 'A_2', 'A_1', 'B_1',
            'A_1', 'A_2', 'B_1', 'B_2', 'A_1', 'A_1', 'A_2', 'B_2', 'B_1',
            'A_1', 'B_2', 'A_2', 'B_2', 'A_2', 'A_2', 'A_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'A_1', 'A_2', 'A_1', 'B_2', 'A_1', 'B_1', 'A_2',
            'B_2', 'B_2', 'A_1', 'A_1', 'B_1', 'B_1', 'B_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'B_2', 'B_1', 'B_1'
        ]
        shuffle = False

        test_ml_train = RunML(
            results_dir, fluor_data, classes, subclasses, shuffle, True
        )

        # Test grid search with PCA
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import recall_score, make_scorer

        parameters =  {'C': np.logspace(-2, 3, num=6, base=10),
                       'gamma': np.logspace(-4, 1, num=6, base=10)}
        splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=1).split(
            X=test_ml_train.x, y=test_ml_train.y, groups=test_ml_train.sub_classes
        ))
        scoring_metric = make_scorer(recall_score, average='macro')

        exp_search_results = {
            'params': [{'SVC__C': 0.01, 'SVC__gamma': 0.0001},
                       {'SVC__C': 0.01, 'SVC__gamma': 0.001},
                       {'SVC__C': 0.01, 'SVC__gamma': 0.01},
                       {'SVC__C': 0.01, 'SVC__gamma': 0.1},
                       {'SVC__C': 0.01, 'SVC__gamma': 1.0},
                       {'SVC__C': 0.01, 'SVC__gamma': 10.0},
                       {'SVC__C': 0.1, 'SVC__gamma': 0.0001},
                       {'SVC__C': 0.1, 'SVC__gamma': 0.001},
                       {'SVC__C': 0.1, 'SVC__gamma': 0.01},
                       {'SVC__C': 0.1, 'SVC__gamma': 0.1},
                       {'SVC__C': 0.1, 'SVC__gamma': 1.0},
                       {'SVC__C': 0.1, 'SVC__gamma': 10.0},
                       {'SVC__C': 1.0, 'SVC__gamma': 0.0001},
                       {'SVC__C': 1.0, 'SVC__gamma': 0.001},
                       {'SVC__C': 1.0, 'SVC__gamma': 0.01},
                       {'SVC__C': 1.0, 'SVC__gamma': 0.1},
                       {'SVC__C': 1.0, 'SVC__gamma': 1.0},
                       {'SVC__C': 1.0, 'SVC__gamma': 10.0},
                       {'SVC__C': 10.0, 'SVC__gamma': 0.0001},
                       {'SVC__C': 10.0, 'SVC__gamma': 0.001},
                       {'SVC__C': 10.0, 'SVC__gamma': 0.01},
                       {'SVC__C': 10.0, 'SVC__gamma': 0.1},
                       {'SVC__C': 10.0, 'SVC__gamma': 1.0},
                       {'SVC__C': 10.0, 'SVC__gamma': 10.0},
                       {'SVC__C': 100.0, 'SVC__gamma': 0.0001},
                       {'SVC__C': 100.0, 'SVC__gamma': 0.001},
                       {'SVC__C': 100.0, 'SVC__gamma': 0.01},
                       {'SVC__C': 100.0, 'SVC__gamma': 0.1},
                       {'SVC__C': 100.0, 'SVC__gamma': 1.0},
                       {'SVC__C': 100.0, 'SVC__gamma': 10.0},
                       {'SVC__C': 1000.0, 'SVC__gamma': 0.0001},
                       {'SVC__C': 1000.0, 'SVC__gamma': 0.001},
                       {'SVC__C': 1000.0, 'SVC__gamma': 0.01},
                       {'SVC__C': 1000.0, 'SVC__gamma': 0.1},
                       {'SVC__C': 1000.0, 'SVC__gamma': 1.0},
                       {'SVC__C': 1000.0, 'SVC__gamma': 10.0}],
            'split0_test_score': np.array([
                0.51190476, 0.51190476, 0.51190476, 0.51190476, 0.66666667,
                0.45238095, 0.51190476, 0.51190476, 0.51190476, 0.51190476,
                0.66666667, 0.45238095, 0.51190476, 0.51190476, 0.51190476,
                0.51190476, 0.52380952, 0.53571429, 0.51190476, 0.51190476,
                0.51190476, 0.58333333, 0.53571429, 0.63095238, 0.51190476,
                0.51190476, 0.51190476, 0.53571429, 0.53571429, 0.64285714,
                0.51190476, 0.51190476, 0.58333333, 0.53571429, 0.53571429,
                0.55952381
            ]),
            'split1_test_score': np.array([
                0.21428571, 0.21428571, 0.14285714, 0.14285714, 0.1547619,
                0.42857143, 0.21428571, 0.21428571, 0.14285714, 0.14285714,
                0.1547619, 0.42857143, 0.21428571, 0.21428571, 0.14285714,
                0.14285714, 0.1547619, 0.60714286, 0.21428571, 0.21428571,
                0.14285714, 0.1547619, 0.28571429, 0.60714286, 0.21428571,
                0.21428571, 0.07142857, 0.08333333, 0.45238095, 0.53571429,
                0.21428571, 0.14285714, 0.1547619, 0.08333333, 0.51190476,
                0.53571429
            ]),
            'split2_test_score': np.array([
                0.17142857, 0.17142857, 0.17142857, 0.31428571, 0.48571429,
                0.51428571, 0.17142857, 0.17142857, 0.17142857, 0.31428571,
                0.48571429, 0.51428571, 0.17142857, 0.17142857, 0.17142857,
                0.31428571, 0.51428571, 0.51428571, 0.17142857, 0.17142857,
                0.17142857, 0.45714286, 0.51428571, 0.51428571, 0.17142857,
                0.17142857, 0.24285714, 0.51428571, 0.51428571, 0.51428571,
                0.17142857, 0.17142857, 0.45714286, 0.51428571, 0.58571429,
                0.58571429
            ]),
            'split3_test_score': np.array([
                0.5, 0.5, 0.41666667, 0.5, 0.66666667, 0.41666667, 0.5, 0.5,
                0.41666667, 0.5, 0.66666667, 0.41666667, 0.5, 0.5, 0.41666667,
                0.5, 0.66666667, 0.41666667, 0.5, 0.5, 0.41666667, 0.5,
                0.58333333, 0.33333333, 0.5, 0.5, 0.41666667, 0.66666667, 0.5,
                0.41666667, 0.5, 0.5, 0.5, 0.58333333, 0.5, 0.33333333
            ]),
            'mean_test_score': np.array([
                0.34940476, 0.34940476, 0.31071429, 0.3672619, 0.49345238,
                0.45297619, 0.34940476, 0.34940476, 0.31071429, 0.3672619,
                0.49345238, 0.45297619, 0.34940476, 0.34940476, 0.31071429,
                0.3672619, 0.46488095, 0.51845238, 0.34940476, 0.34940476,
                0.31071429, 0.42380952, 0.4797619 , 0.52142857, 0.34940476,
                0.34940476, 0.31071429, 0.45, 0.50059524, 0.52738095,
                0.34940476, 0.33154762, 0.42380952, 0.42916667, 0.53333333,
                0.50357143
            ]),
            'std_test_score': np.array([
                0.15733552, 0.15733552, 0.15754368, 0.15141411, 0.20903199,
                0.03766028, 0.15733552, 0.15733552, 0.15754368, 0.15141411,
                0.20903199, 0.03766028, 0.15733552, 0.15733552, 0.15754368,
                0.15141411, 0.18894755, 0.06808389, 0.15733552, 0.15733552,
                0.15754368, 0.16182596, 0.11478685, 0.11701842, 0.15733552,
                0.15733552, 0.1684976, 0.21958397, 0.03060126, 0.08036376,
                0.15733552, 0.17474775, 0.16182596, 0.20122481, 0.03286232,
                0.09986527
            ]),
            'rank_test_score': np.array(
                [20, 20, 32, 17, 7, 11, 20, 20, 32, 17, 7, 11, 20, 20, 32, 17,
                 10, 4, 20, 20, 32, 15, 9, 3, 20, 20, 32, 13, 6, 2, 20, 31, 15,
                 14, 1, 5], dtype=np.int32)
        }
        exp_best_params = {'SVC__C': 1000.0,
                           'SVC__gamma': 1.0}

        act_grid_search = test_ml_train.run_grid_search(
            test_ml_train.x, test_ml_train.y, test_ml_train.sub_classes,
            ['Feature_1', 'Feature_2', 'Feature_3'], SVC(random_state=1),
            splits, 'smotetomek', 1, parameters, scoring_metric, True
        )
        act_search_results = act_grid_search.cv_results_
        for key in [
            'mean_fit_time', 'std_fit_time', 'mean_score_time',
            'std_score_time', 'param_SVC__C', 'param_SVC__gamma'
        ]:
            del(act_search_results[key])
        act_best_params = act_grid_search.best_params_

        self.assertEqual(list(exp_search_results.keys()),
                         list(act_search_results.keys()))
        for key in exp_search_results.keys():
            if type(exp_search_results[key]) == np.ndarray:
                np.testing.assert_almost_equal(
                    exp_search_results[key], act_search_results[key], 7
                )
            else:
                self.assertEqual(
                    exp_search_results[key], act_search_results[key]
                )
        self.assertEqual(exp_best_params, act_best_params)

        # Test grid search without PCA
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import GroupKFold

        splits = list(GroupKFold(n_splits=2).split(
            X=test_ml_train.x, y=test_ml_train.y,
            groups=test_ml_train.sub_classes
        ))

        exp_search_results = {
            'params': [{}],
            'split0_test_score': np.array([0.52]),
            'split1_test_score': np.array([0.4]),
            'mean_test_score': np.array([0.46]),
            'std_test_score': np.array([0.06]),
            'rank_test_score': np.array([1], dtype=np.int32)
        }
        exp_best_params = {}

        act_grid_search = test_ml_train.run_grid_search(
            test_ml_train.x, test_ml_train.y, test_ml_train.sub_classes,
            ['Feature_1', 'Feature_2'], GaussianNB(), splits, 'no_balancing',
            None, {}, 'accuracy', True
        )
        act_search_results = act_grid_search.cv_results_
        for key in [
            'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'
        ]:
            del(act_search_results[key])
        act_best_params = act_grid_search.best_params_

        self.assertEqual(list(exp_search_results.keys()),
                         list(act_search_results.keys()))
        for key in exp_search_results.keys():
            if type(exp_search_results[key]) == np.ndarray:
                np.testing.assert_almost_equal(
                    exp_search_results[key], act_search_results[key], 7
                )
            else:
                self.assertEqual(
                    exp_search_results[key], act_search_results[key]
                )
        self.assertEqual(exp_best_params, act_best_params)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_test_model(self):
        """
        Tests test_model in train.py
        """

        print('Testing test_model')

        results_dir = 'tests/Temp_output'
        x_train = pd.DataFrame({
            'Feature_1': [5, 9, 8, 1, 3, 5, 10, 6, 7, 1, 8, 9, 1, 10, 2, 2, 8,
                          7, 1, 3, 8, 4, 3, 4, 4, 6, 2, 10, 4, 5, 1, 7, 10, 3,
                          10, 6, 3, 8, 1, 4, 6, 1, 5, 2, 2, 1, 7, 1, 2, 4],
            'Feature_2': [9, 9, 7, 9, 6, 4, 7, 4, 2, 9, 7, 9, 7, 6, 4, 10, 8, 1,
                          5, 4, 3, 3, 4, 3, 1, 4, 9, 6, 7, 10, 4, 6, 9, 2, 7, 4,
                          3, 5, 7, 10, 1, 5, 3, 7, 2, 5, 10, 2, 2, 5],
            'Feature_3': [5, 4, 8, 10, 3, 2, 10, 5, 1, 10, 5, 5, 5, 10, 7, 1, 8,
                          8, 2, 1, 10, 9, 10, 6, 7, 4, 3, 3, 10, 10, 4, 7, 4, 6,
                          10, 7, 6, 9, 4, 9, 9, 4, 4, 5, 10, 2, 10, 1, 7, 10]
        })
        y_train = [
            'A', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'B',
            'A', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'A',
            'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'A',
            'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'B'
        ]
        y_groups = [
            'A_1', 'B_2', 'B_1', 'A_2', 'B_1', 'B_2', 'A_2', 'A_1', 'B_1',
            'A_1', 'A_2', 'B_1', 'B_2', 'A_1', 'A_1', 'A_2', 'B_2', 'B_1',
            'A_1', 'B_2', 'A_2', 'B_2', 'A_2', 'A_2', 'A_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'A_1', 'A_2', 'A_1', 'B_2', 'A_1', 'B_1', 'A_2',
            'B_2', 'B_2', 'A_1', 'A_1', 'B_1', 'B_1', 'B_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'B_2', 'B_1', 'B_1'
        ]
        x_test = pd.DataFrame({
            'Feature_1': [8, 6, 1, 7, 8, 6, 7, 9, 1, 1, 7, 5, 4, 4, 1, 2, 2, 3, 1, 8],
            'Feature_2': [8, 8, 10, 6, 5, 7, 8, 10, 8, 1, 1, 10, 1, 9, 1, 8, 10, 3, 9, 4],
            'Feature_3': [7, 6, 3, 1, 4, 4, 9, 8, 3, 8, 9, 3, 10, 3, 7, 4, 4, 1, 5, 10]
        })
        y_test = np.array(
            ['B', 'A', 'A', 'B', 'A', 'A', 'A', 'B', 'B', 'B',
             'B', 'A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B']
        )
        shuffle = False

        test_ml_train = RunML(
            results_dir, x_train, y_train, y_groups, shuffle, True
        )

        # Define function arguments
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            cohen_kappa_score
        )
        trained_clf = AdaBoostClassifier(n_estimators=100, random_state=1).fit(
            test_ml_train.x, test_ml_train.y
        )
        test_scoring_funcs = OrderedDict({
            accuracy_score: {},
            f1_score: {'average': 'weighted'},
            precision_score: {'pos_label': 'B'},
            recall_score: {'average': 'macro'},
            cohen_kappa_score: {'weights': 'quadratic'}
        })

        # Test test_model
        exp_predictions = np.array(
            ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'B',
             'B', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'A']
        )
        exp_test_scores = OrderedDict({
            'accuracy': 0.55,
            'f1': 0.5488721804511277,
            'precision': 0.5454545454545454,
            'recall': 0.55,
            'cohen_kappa': 0.09999999999999998
        })
        act_predictions, act_test_scores = test_ml_train.test_model(
            x_test.to_numpy(), y_test, trained_clf, test_scoring_funcs, False,
            '', True
        )

        np.testing.assert_equal(exp_predictions, act_predictions)
        self.assertEqual(list(exp_test_scores.keys()),
                         list(act_test_scores.keys()))
        for key in exp_test_scores.keys():
            exp_val = exp_test_scores[key]
            act_val = act_test_scores[key]
            np.testing.assert_almost_equal(exp_val, act_val, 7)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_run_ml(self):
        """
        Tests run_ml in train.py
        """

        print('Testing run_ml')

        results_dir = 'tests/Temp_output'
        x_train = pd.DataFrame({
            'Feature_1': [5, 9, 8, 1, 3, 5, 10, 6, 7, 1, 8, 9, 1, 10, 2, 2, 8,
                          7, 1, 3, 8, 4, 3, 4, 4, 6, 2, 10, 4, 5, 1, 7, 10, 3,
                          10, 6, 3, 8, 1, 4, 6, 1, 5, 2, 2, 1, 7, 1, 2, 4],
            'Feature_2': [9, 9, 7, 9, 6, 4, 7, 4, 2, 9, 7, 9, 7, 6, 4, 10, 8, 1,
                          5, 4, 3, 3, 4, 3, 1, 4, 9, 6, 7, 10, 4, 6, 9, 2, 7, 4,
                          3, 5, 7, 10, 1, 5, 3, 7, 2, 5, 10, 2, 2, 5],
            'Feature_3': [5, 4, 8, 10, 3, 2, 10, 5, 1, 10, 5, 5, 5, 10, 7, 1, 8,
                          8, 2, 1, 10, 9, 10, 6, 7, 4, 3, 3, 10, 10, 4, 7, 4, 6,
                          10, 7, 6, 9, 4, 9, 9, 4, 4, 5, 10, 2, 10, 1, 7, 10]
        })
        y_train = [
            'A', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'B',
            'A', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'A',
            'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'A',
            'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'B'
        ]
        y_groups = [
            'A_1', 'B_2', 'B_1', 'A_2', 'B_1', 'B_2', 'A_2', 'A_1', 'B_1',
            'A_1', 'A_2', 'B_1', 'B_2', 'A_1', 'A_1', 'A_2', 'B_2', 'B_1',
            'A_1', 'B_2', 'A_2', 'B_2', 'A_2', 'A_2', 'A_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'A_1', 'A_2', 'A_1', 'B_2', 'A_1', 'B_1', 'A_2',
            'B_2', 'B_2', 'A_1', 'A_1', 'B_1', 'B_1', 'B_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'B_2', 'B_1', 'B_1'
        ]
        x_test = pd.DataFrame({
            'Feature_1': [8, 6, 1, 7, 8, 6, 7, 9, 1, 1, 7, 5, 4, 4, 1, 2, 2, 3, 1, 8],
            'Feature_2': [8, 8, 10, 6, 5, 7, 8, 10, 8, 1, 1, 10, 1, 9, 1, 8, 10, 3, 9, 4],
            'Feature_3': [7, 6, 3, 1, 4, 4, 9, 8, 3, 8, 9, 3, 10, 3, 7, 4, 4, 1, 5, 10]
        }).to_numpy()
        y_test = np.array(
            ['B', 'A', 'A', 'B', 'A', 'A', 'A', 'B', 'B', 'B',
             'B', 'A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B']
        )
        shuffle = False

        test_ml_train = RunML(
            results_dir, x_train, y_train, y_groups, shuffle, True
        )

        # Define function arguments
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            cohen_kappa_score, make_scorer
        )

        splits = list(StratifiedKFold(
            n_splits=2, shuffle=True, random_state=1
        ).split(X=x_train, y=y_train))
        f1 = make_scorer(f1_score, average='binary', pos_label='A')
        precision = make_scorer(precision_score, average='weighted')
        test_scoring_funcs = OrderedDict({
            accuracy_score: {},
            f1_score: {'average': 'weighted'},
            precision_score: {'pos_label': 'B'},
            recall_score: {'average': 'macro'},
            cohen_kappa_score: {'weights': 'quadratic'}
        })

        # Test run_ml for randomised search
        exp_search_results = {
            'params': [{'AdaBoostClassifier__n_estimators': 100},
                       {'AdaBoostClassifier__n_estimators': 30},
                       {'AdaBoostClassifier__n_estimators': 1000},
                       {'AdaBoostClassifier__n_estimators': 10}],
            'split0_test_score': np.array([0., 0., 0., 0.]),
            'split1_test_score': np.array([0.55172414, 0.55172414, 0.55172414, 0.55172414]),
            'mean_test_score': np.array([0.27586207, 0.27586207, 0.27586207, 0.27586207]),
            'std_test_score': np.array([0.27586207, 0.27586207, 0.27586207, 0.27586207]),
            'rank_test_score': np.array([1, 1, 1, 1], dtype=np.int32)
        }
        exp_best_params = {'AdaBoostClassifier__n_estimators': 100}

        act_random_search = test_ml_train.run_ml(
            AdaBoostClassifier(random_state=1), x_train=test_ml_train.x,
            y_train=test_ml_train.y, train_groups=test_ml_train.sub_classes,
            x_test=x_test, y_test=y_test,
            selected_features=['Feature_2', 'Feature_3'], splits=None,
            resampling_method='smoteenn', n_components_pca=2,
            run='randomsearch',
            params={'n_estimators': np.array([10, 30, 100, 300, 1000])},
            train_scoring_metric=f1, test_scoring_funcs={}, n_iter=4,
            cv_folds=2, test=True
        )
        act_search_results = act_random_search.cv_results_
        for key in [
            'mean_fit_time', 'std_fit_time', 'mean_score_time',
            'std_score_time', 'param_AdaBoostClassifier__n_estimators'
        ]:
            del(act_search_results[key])
        act_best_params = act_random_search.best_params_

        self.assertEqual(list(exp_search_results.keys()),
                         list(act_search_results.keys()))
        for key in exp_search_results.keys():
            if type(exp_search_results[key]) == np.ndarray:
                np.testing.assert_almost_equal(
                    exp_search_results[key], act_search_results[key], 7
                )
            else:
                self.assertEqual(
                    exp_search_results[key], act_search_results[key]
                )
        self.assertEqual(exp_best_params, act_best_params)

        # Test run_ml for grid search
        exp_search_results = {
            'params': [{'AdaBoostClassifier__n_estimators': 10},
                       {'AdaBoostClassifier__n_estimators': 30},
                       {'AdaBoostClassifier__n_estimators': 100},
                       {'AdaBoostClassifier__n_estimators': 300},
                       {'AdaBoostClassifier__n_estimators': 1000}],
            'split0_test_score': np.array([0.58655462, 0.47385621, 0.41176471, 0.52406417, 0.52406417]),
            'split1_test_score': np.array([0.48190045, 0.31334842, 0.40784314, 0.47860963, 0.47860963]),
            'split2_test_score': np.array([0.375     , 0.375     , 0.42708333, 0.42708333, 0.42708333]),
            'mean_test_score': np.array([0.48115169, 0.38740154, 0.41556373, 0.47658571, 0.47658571]),
            'std_test_score': np.array([0.08636844, 0.0661112 , 0.00830143, 0.03961812, 0.03961812]),
            'rank_test_score': np.array([1, 5, 4, 2, 2], dtype=np.int32)
        }
        exp_best_params = {'AdaBoostClassifier__n_estimators': 10}

        act_grid_search = test_ml_train.run_ml(
            AdaBoostClassifier(random_state=1), x_train=test_ml_train.x,
            y_train=test_ml_train.y, train_groups=None, x_test=x_test,
            y_test=y_test, selected_features=['Feature_2', 'Feature_1'],
            splits=None, resampling_method='no_balancing',
            n_components_pca=None, run='gridsearch',
            params={'n_estimators': np.array([10, 30, 100, 300, 1000])},
            train_scoring_metric=precision, test_scoring_funcs={},
            n_iter=3, cv_folds=3, test=True
        )
        act_search_results = act_grid_search.cv_results_
        for key in [
            'mean_fit_time', 'std_fit_time', 'mean_score_time',
            'std_score_time', 'param_AdaBoostClassifier__n_estimators'
        ]:
            del(act_search_results[key])
        act_best_params = act_grid_search.best_params_

        self.assertEqual(list(exp_search_results.keys()),
                         list(act_search_results.keys()))
        for key in exp_search_results.keys():
            if type(exp_search_results[key]) == np.ndarray:
                np.testing.assert_almost_equal(
                    exp_search_results[key], act_search_results[key], 7
                )
            else:
                self.assertEqual(
                    exp_search_results[key], act_search_results[key]
                )
        self.assertEqual(exp_best_params, act_best_params)

        # Test run_ml for training ML model
        exp_predictions = np.array(
            ['B', 'B', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'B',
             'A', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'A', 'B']
        )
        exp_test_scores = OrderedDict({
            'accuracy': 0.6,
            'f1': 0.595959595959596,
            'precision': 0.625,
            'recall': 0.6,
            'cohen_kappa': 0.19999999999999996
        })
        act_trained_clf, act_predictions, act_test_scores = test_ml_train.run_ml(
            AdaBoostClassifier, x_train=test_ml_train.x,
            y_train=test_ml_train.y, train_groups=test_ml_train.sub_classes,
            x_test=x_test, y_test=y_test,
            selected_features=['Feature_2', 'Feature_3', 'Feature_1'],
            splits=splits, resampling_method='max_sampling', n_components_pca=2,
            run='train', params={'n_estimators': 100, 'random_state': 1},
            train_scoring_metric='accuracy',
            test_scoring_funcs=test_scoring_funcs, n_iter=None, cv_folds=5,
            test=True
        )

        np.testing.assert_equal(exp_predictions, act_predictions)
        self.assertEqual(list(exp_test_scores.keys()),
                         list(act_test_scores.keys()))
        for key in exp_test_scores.keys():
            exp_val = exp_test_scores[key]
            act_val = act_test_scores[key]
            np.testing.assert_almost_equal(exp_val, act_val, 7)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')

    def test_run_nested_CV(self):
        """
        Tests run_nested_CV in train.py
        """

        print('Testing run_nested_CV')

        # Removes directory created by defining RunML object
        #shutil.rmtree('tests/Temp_output')

    def test_run_5x2_CV_paired_t_test(self):
        """
        Tests run_5x2_CV_paired_t_test in train.py
        """

        print('Testing run_5x2_CV_paired_t_test')

        results_dir = 'tests/Temp_output'
        x_train = pd.DataFrame({
            'Feature_1': [5, 9, 8, 1, 3, 5, 10, 6, 7, 1, 8, 9, 1, 10, 2, 2, 8,
                          7, 1, 3, 8, 4, 3, 4, 4, 6, 2, 10, 4, 5, 1, 7, 10, 3,
                          10, 6, 3, 8, 1, 4, 6, 1, 5, 2, 2, 1, 7, 1, 2, 4],
            'Feature_2': [9, 9, 7, 9, 6, 4, 7, 4, 2, 9, 7, 9, 7, 6, 4, 10, 8, 1,
                          5, 4, 3, 3, 4, 3, 1, 4, 9, 6, 7, 10, 4, 6, 9, 2, 7, 4,
                          3, 5, 7, 10, 1, 5, 3, 7, 2, 5, 10, 2, 2, 5],
            'Feature_3': [5, 4, 8, 10, 3, 2, 10, 5, 1, 10, 5, 5, 5, 10, 7, 1, 8,
                          8, 2, 1, 10, 9, 10, 6, 7, 4, 3, 3, 10, 10, 4, 7, 4, 6,
                          10, 7, 6, 9, 4, 9, 9, 4, 4, 5, 10, 2, 10, 1, 7, 10]
        })
        y_train = [
            'A', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'B',
            'A', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'A',
            'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'A',
            'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'B'
        ]
        y_groups = [
            'A_1', 'B_2', 'B_1', 'A_2', 'B_1', 'B_2', 'A_2', 'A_1', 'B_1',
            'A_1', 'A_2', 'B_1', 'B_2', 'A_1', 'A_1', 'A_2', 'B_2', 'B_1',
            'A_1', 'B_2', 'A_2', 'B_2', 'A_2', 'A_2', 'A_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'A_1', 'A_2', 'A_1', 'B_2', 'A_1', 'B_1', 'A_2',
            'B_2', 'B_2', 'A_1', 'A_1', 'B_1', 'B_1', 'B_2', 'A_1', 'A_1',
            'A_2', 'B_2', 'B_2', 'B_1', 'B_1'
        ]
        shuffle = False

        test_ml_train = RunML(
            results_dir, x_train, y_train, y_groups, shuffle, True
        )

        # Define function arguments
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier

        # Test arguments
        exp_F = 0.7999999999999998
        exp_p = 0.6439069510127531
        act_F, act_p = test_ml_train.run_5x2_CV_paired_t_test(
            x_train.to_numpy()[0:40,:], np.array(y_train)[0:40],
            ['Feature_2', 'Feature_3'], ['Feature_2', 'Feature_3'], SVC,
            RandomForestClassifier, {'C': 10, 'gamma': 0.01, 'random_state': 1},
            {'n_estimators': 250, 'min_samples_split': 5, 'min_samples_leaf': 2,
             'random_state': 1}, 'max_sampling', 'no_balancing', 2, None, True
        )
        np.testing.assert_almost_equal(exp_F, act_F, 7)
        np.testing.assert_almost_equal(exp_p, act_p, 7)

        # Removes directory created by defining RunML object
        shutil.rmtree('tests/Temp_output')
