
# python -m unittest tests/test_array_sensing.py

import numpy as np
import os
import pandas as pd
import pickle
import shutil
import unittest
from collections import OrderedDict
from sklearn.preprocessing import RobustScaler
from subroutines.exceptions import NaNFluorescenceError, PlateLayoutError
from subroutines.parse_array_data import (
    trim_dataframe, parse_xlsx_to_dataframe, draw_scatter_plot,
    check_for_saturation, highlight_outliers, calc_median, scale_min_max,
    draw_boxplots, draw_heatmap_fingerprints
)


class TestClass(unittest.TestCase):

    def test_trim_dataframe(self):
        """
        Tests trim_dataframe in parse_array_data
        """

        print('Testing trim_dataframe')

        exp_dfs = {'Test_dataframe_1.pkl': pd.DataFrame(np.array([[1.0, 2.0, 3.0],
                                                                  [4.0, 5.0, 6.0],
                                                                  [7.0, 8.0, 9.0]])),
                   'Test_dataframe_2.pkl': pd.DataFrame(np.array([[1.0, 2.0, 3.0],
                                                                  [4.0, 5.0, 6.0]])),
                   'Test_dataframe_3.pkl': pd.DataFrame(np.array([[1.0, 2.0],
                                                                  [4.0, 5.0],
                                                                  [7.0, 8.0]])),
                   'Test_dataframe_4.pkl': pd.DataFrame(np.array([[1.0],
                                                                  [4.0],
                                                                  [7.0]])),
                   'Test_dataframe_5.pkl': pd.DataFrame(np.array([[1.0, 2.0, 3.0]])),
                   'Test_dataframe_6.pkl': pd.DataFrame(np.array([[1.0, np.nan, 3.0],
                                                                  [np.nan, np.nan, 6.0],
                                                                  [7.0, 8.0, 9.0]])),
                   'Test_dataframe_7.pkl': pd.DataFrame(np.array([['1', 2.0]], dtype=object)),
                   'Test_dataframe_8.pkl': np.nan,
                   'Test_dataframe_9.pkl': np.nan,
                   'Test_dataframe_10.pkl': np.nan}

        for df_file, exp_df in exp_dfs.items():
            num = df_file.split('_')[2].split('.')[0]
            df = pd.read_pickle('tests/Test_dataframes/{}'.format(df_file))

            if num in ['1', '2', '3', '4', '5', '6', '7']:
                obs_df = trim_dataframe(df, df_file)
                pd.testing.assert_frame_equal(exp_df, obs_df)
            elif num in ['8', '9', '10']:
                with self.assertRaises(PlateLayoutError): trim_dataframe(df, df_file)

    def test_parse_xlsx_to_dataframe(self):
        """
        Tests parse_xlsx_to_dataframe in parse_array_data
        """

        print('Testing parse_xlsx_to_dataframe')

        """
        exp_dfs = {}

        for excel_file, exp_df in exp_dfs.items():
            num = excel_file.split('_')[2].split('.')[0]
            with self.assertRaises(TypeError): parse_xlsx_to_dataframe(excel_file, split, peptide_dict, gain='a')
        """

    def test_draw_scatter_plot(self):
        """
        Tests draw_scatter_plot in parse_array_data
        """

        print('Testing draw_scatter_plot')

        exp_input_dict = {
            1: [{'Analyte 1': pd.DataFrame({'Peptide 1': [1.0, 2.0],
                                            'Peptide 2': [2.0, np.nan],
                                            'Peptide 3': [np.nan, np.nan]}),
                 'Analyte 2': pd.DataFrame({'Peptide 1': [1.5, 2.5],
                                            'Peptide 2': [3.0, np.nan],
                                            'Peptide 3': [1.0, 3.0]})},
                ['Peptide 1', 'Peptide 2', 'Peptide 3'],
                ['Analyte 1', 'Analyte 2']],
            2: [{'Analyte 1': pd.DataFrame({'Peptide 1': [1.0, 2.0],
                                            'Peptide 2': [2.0, np.nan],
                                            'Peptide 3': [np.nan, np.nan]}),
                 'Analyte 2': pd.DataFrame({'Peptide 1': [1.5, 2.5],
                                            'Peptide 2': [3.0, np.nan],
                                            'Peptide 3': [1.0, 3.0]})},
                ['Peptide 2'],
                ['Analyte 1']],
            3: [{'Analyte 1': pd.DataFrame({'Peptide 1': [1.0, 2.0],
                                            'Peptide 2': [2.0, np.nan],
                                            'Peptide 3': [np.nan, np.nan]}),
                 'Analyte 2': pd.DataFrame({'Peptide 1': [1.5, 2.5],
                                            'Peptide 2': [3.0, np.nan],
                                            'Peptide 3': [1.0, 3.0]})},
                [],
                ['Analyte 1']],
            4: [{'Analyte 1': pd.DataFrame({'Peptide 1': [1.0, 2.0],
                                            'Peptide 2': [2.0, np.nan],
                                            'Peptide 3': [np.nan, np.nan]}),
                 'Analyte 2': pd.DataFrame({'Peptide 1': [1.5, 2.5],
                                            'Peptide 2': [3.0, np.nan],
                                            'Peptide 3': [1.0, 3.0]})},
                ['Peptide 1'],
                []],
            5: [{'Analyte 1': pd.DataFrame({'Peptide 1': [1.0, 2.0],
                                            'Peptide 2': [2.0, np.nan],
                                            'Peptide 3': [np.nan, np.nan]}),
                 'Analyte 2': pd.DataFrame({'Peptide 1': [1.5, 2.5],
                                            'Peptide 2': [3.0, np.nan],
                                            'Peptide 3': [1.0, 3.0]})},
                ['Peptide 1', 'Peptide 4'],
                ['Analyte 1']],
            6: [{'Analyte 1': pd.DataFrame({'Peptide 1': [1.0, 2.0],
                                            'Peptide 2': [2.0, np.nan],
                                            'Peptide 3': [np.nan, np.nan]}),
                 'Analyte 2': pd.DataFrame({'Peptide 1': [1.5, 2.5],
                                            'Peptide 2': [3.0, np.nan],
                                            'Peptide 3': [1.0, 3.0]})},
                ['Peptide 1'],
                ['Analyte 1', 'Analyte 4']],
            7: [{'Analyte 1': pd.DataFrame({'Peptide 1': [1.0, 2.0],
                                            'Peptide 2': ['X', np.nan],
                                            'Peptide 3': [np.nan, np.nan]}),
                 'Analyte 2': pd.DataFrame({'Peptide 1': [1.5, 2.5],
                                            'Peptide 2': [3.0, np.nan],
                                            'Peptide 3': [1.0, 3.0]})},
                ['Peptide 1', 'Peptide 2', 'Peptide 3'],
                ['Analyte 1', 'Analyte 2']],
            8: [{'Analyte 2_10': pd.DataFrame({'Peptide 1': [1.0, 2.0],
                                               'Peptide 2': [2.0, np.nan],
                                               'Peptide 3': [np.nan, np.nan]}),
                 'Analyte 2_1': pd.DataFrame({'Peptide 1': [1.5, 2.5],
                                              'Peptide 2': [3.0, np.nan],
                                              'Peptide 3': [1.0, 3.0]}),
                 'Analyte 2_1.5': pd.DataFrame({'Peptide 1': [3.0, 2.5],
                                                'Peptide 2': [0.5, 1.5],
                                                'Peptide 3': [np.nan, 2.5]}),
                 'Analyte 2_Z': pd.DataFrame({'Peptide 1': [np.nan, np.nan],
                                              'Peptide 2': [0.5, 2.0],
                                              'Peptide 3': [1.5, 1.0]})},
                ['Peptide 1', 'Peptide 2', 'Peptide 3'],
                ['Analyte 2_10', 'Analyte 2_1', 'Analyte 2_1.5', 'Analyte 2_Z']],
        }
        exp_results_dict = {
            1: [pd.DataFrame({'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 2', 'Analyte 2', 'Analyte 1', 'Analyte 1', 'Analyte 2', 'Analyte 2', 'Analyte 1', 'Analyte 1', 'Analyte 2', 'Analyte 2'],
                              'Peptide': ['Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 3', 'Peptide 3', 'Peptide 3', 'Peptide 3'],
                              'Reading': [1.0, 2.0, 1.5, 2.5, 2.0, np.nan, 3.0, np.nan, np.nan, np.nan, 1.0, 3.0]}),
                ['Analyte 1', 'Analyte 2']],
            2: [pd.DataFrame({'Analyte': ['Analyte 1', 'Analyte 1'],
                              'Peptide': ['Peptide 2', 'Peptide 2'],
                              'Reading': [2.0, np.nan]}),
                ['Analyte 1']],
            3: [pd.DataFrame({}), []],
            4: [pd.DataFrame({}), []],
            5: [pd.DataFrame({}), []],
            6: [pd.DataFrame({}), []],
            7: [pd.DataFrame({}), []],
            8: [pd.DataFrame({'Analyte': ['Analyte 2_10', 'Analyte 2_10', 'Analyte 2_1', 'Analyte 2_1', 'Analyte 2_1.5', 'Analyte 2_1.5', 'Analyte 2_Z', 'Analyte 2_Z', 'Analyte 2_10', 'Analyte 2_10', 'Analyte 2_1', 'Analyte 2_1', 'Analyte 2_1.5', 'Analyte 2_1.5', 'Analyte 2_Z', 'Analyte 2_Z', 'Analyte 2_10', 'Analyte 2_10', 'Analyte 2_1', 'Analyte 2_1', 'Analyte 2_1.5', 'Analyte 2_1.5', 'Analyte 2_Z', 'Analyte 2_Z'],
                              'Peptide': ['Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 3', 'Peptide 3', 'Peptide 3', 'Peptide 3', 'Peptide 3', 'Peptide 3', 'Peptide 3', 'Peptide 3'],
                              'Reading': [1.0, 2.0, 1.5, 2.5, 3.0, 2.5, np.nan, np.nan, 2.0, np.nan, 3.0, np.nan, 0.5, 1.5, 0.5, 2.0, np.nan, np.nan, 1.0, 3.0, np.nan, 2.5, 1.5, 1.0]}),
                ['Analyte 2_Z', 'Analyte 2_1', 'Analyte 2_1.5', 'Analyte 2_10']],
        }

        for num in exp_input_dict.keys():
            test_data = exp_input_dict[num][0]
            features = exp_input_dict[num][1]
            analytes = exp_input_dict[num][2]
            exp_df = exp_results_dict[num][0]
            exp_analytes_list = exp_results_dict[num][1]

            os.mkdir('tests/Temp_output')

            if num in [3, 4, 5, 6, 7]:
                with self.assertRaises(ValueError): draw_scatter_plot(
                    test_data, 'tests/Temp_output', features, analytes, str(num), True
                )
            else:
                draw_scatter_plot(
                    test_data, 'tests/Temp_output', features, analytes, str(num), True
                )
                act_df = pd.read_pickle('tests/Temp_output/Plot_data.pkl')
                with open('tests/Temp_output/Sorted_analytes.pkl', 'rb') as f:
                    act_analytes_list = pickle.load(f)
                if not os.path.isfile('tests/Temp_output/{}_data_spread.svg'.format(num)):
                    raise FileNotFoundError(
                        'draw_scatter_plot has failed to create the expected plots '
                        '- expect file tests/Temp_output/{}_data_spread.svg to '
                        'exist'.format(num)
                    )

                pd.testing.assert_frame_equal(exp_df, act_df)
                self.assertEqual(exp_analytes_list, act_analytes_list)

            shutil.rmtree('tests/Temp_output')

    def test_check_for_saturation(self):
        """
        Tests check_for_saturation in parse_array_data
        """

        print('Testing check_for_saturation')

        plate = {'Analyte': pd.DataFrame({'Peptide_1': [1.0, 2.0, 3.0],
                                          'Peptide_2': [4.0, np.inf, np.nan],
                                          'Peptide_3': [1.0, 4.0, 5.0]})}
        exp_readings = [['', 'Analyte', 'Peptide_1', 1.0],
                        ['', 'Analyte', 'Peptide_2', 4.0],
                        ['', 'Analyte', 'Peptide_2', np.inf]]
        obs_readings = check_for_saturation(
            plate, '', ['Peptide_1', 'Peptide_2'], 1.5, 3.5
        )

        self.assertEqual(exp_readings, obs_readings)

    def test_highlight_outliers(self):
        """
        Tests highlight_outliers in parse_array_data
        """

        print('Testing highlight_outliers')

        input_dfs = {1: pd.DataFrame({}),
                     2: pd.DataFrame({'1': [1.0, 2.0, 13.0],
                                      '2': [np.nan, np.nan, np.nan]}),
                     3: pd.DataFrame({'1': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                            14.0, 16.0],
                                      '2': [3.0, 18.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0],
                                      'Analyte': ['a', 'b', 'c', 'd', 'e', 'f',
                                                  'g', 'h', 'i', 'j', 'k', 'l',
                                                  'm', 'n', 'o', 'p', 'q', 'r',
                                                  's', 't', 'u', 'v', 'w', 'x',
                                                  'y', 'z']}),
                     4: pd.DataFrame({'1': [0.0001, 100],
                                      '2': [2, 2],
                                      'Analyte': ['a', 'b']}),
                     5: pd.DataFrame({'1': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                            14.0, 16.0, np.nan],
                                      '2': [3.0, 18.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 4.0],
                                      'Analyte': ['a', 'b', 'c', 'd', 'e', 'f',
                                                  'g', 'h', 'i', 'j', 'k', 'l',
                                                  'm', 'n', 'o', 'p', 'q', 'r',
                                                  's', 't', 'u', 'v', 'w', 'x',
                                                  'y', 'z', 'aa']}),
                     6: pd.DataFrame({'1': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                            1.0, 1.0],
                                      '2': [3.0, 18.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                                            3.0, 4.0],
                                      'Analyte': ['a', 'b', 'c', 'd', 'e', 'f',
                                                  'g', 'h', 'i', 'j', 'k', 'l',
                                                  'm', 'n', 'o', 'p', 'q', 'r',
                                                  's', 't', 'u', 'v', 'w', 'x',
                                                  'y', 'z']})}

        for num, df in input_dfs.items():
            if num in [1, 2]:
                with self.assertRaises(ValueError): highlight_outliers(df, False)

            elif num == 3:
                # Default parameters, don't update input dataframe
                act_outliers = highlight_outliers(df, False)
                exp_outliers = [['z', '1', 16.0], ['y', '1', 14.0],
                                ['b', '2', 18.0]]
                self.assertEqual(exp_outliers, act_outliers)

                # Test user_k_max
                act_outliers = highlight_outliers(df, False, user_k_max=0)
                exp_outliers = []
                self.assertEqual(exp_outliers, act_outliers)

                # Test user_k_max
                act_outliers = highlight_outliers(df, False, user_k_max=1)
                exp_outliers = [['z', '1', 16.0], ['b', '2', 18.0]]
                self.assertEqual(exp_outliers, act_outliers)

                # Test user_k_max
                act_outliers = highlight_outliers(df, False, user_k_max=2)
                exp_outliers = [['z', '1', 16.0], ['y', '1', 14.0],
                                ['b', '2', 18.0]]
                self.assertEqual(exp_outliers, act_outliers)

                # Test user_k_max
                act_outliers = highlight_outliers(df, False, user_k_max=3)
                exp_outliers = [['z', '1', 16.0], ['y', '1', 14.0],
                                ['b', '2', 18.0]]
                self.assertEqual(exp_outliers, act_outliers)

                # Test drop_thresh
                act_df, act_outliers = highlight_outliers(df, True, drop_thresh=1)
                exp_df = pd.DataFrame({
                    '1': [1.0, 1.0, 2.0, 1.0, 2.0,
                          1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                          1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                          1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                    '2': [3.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
                    'Analyte': ['a', 'c', 'd', 'e', 'f',
                                'g', 'h', 'i', 'j', 'k', 'l',
                                'm', 'n', 'o', 'p', 'q', 'r',
                                's', 't', 'u', 'v', 'w', 'x']})
                exp_outliers = OrderedDict({
                    0: [0, []],
                    1: [1, [1]],
                    2: [0, []],
                    3: [0, []],
                    4: [0, []],
                    5: [0, []],
                    6: [0, []],
                    7: [0, []],
                    8: [0, []],
                    9: [0, []],
                    10: [0, []],
                    11: [0, []],
                    12: [0, []],
                    13: [0, []],
                    14: [0, []],
                    15: [0, []],
                    16: [0, []],
                    17: [0, []],
                    18: [0, []],
                    19: [0, []],
                    20: [0, []],
                    21: [0, []],
                    22: [0, []],
                    23: [0, []],
                    24: [1, [0]],
                    25: [1, [0]]
                })
                pd.testing.assert_frame_equal(exp_df, act_df)
                self.assertEqual(exp_outliers, act_outliers)

                # Test drop_thresh
                act_df, act_outliers = highlight_outliers(df, True, drop_thresh=2)
                exp_df = pd.DataFrame({
                    '1': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                          1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                          1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                          1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                          14.0, 16.0],
                    '2': [3.0, 18.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0],
                    'Analyte': ['a', 'b', 'c', 'd', 'e', 'f',
                                'g', 'h', 'i', 'j', 'k', 'l',
                                'm', 'n', 'o', 'p', 'q', 'r',
                                's', 't', 'u', 'v', 'w', 'x',
                                'y', 'z']})
                exp_outliers = OrderedDict({
                    0: [0, []],
                    1: [1, [1]],
                    2: [0, []],
                    3: [0, []],
                    4: [0, []],
                    5: [0, []],
                    6: [0, []],
                    7: [0, []],
                    8: [0, []],
                    9: [0, []],
                    10: [0, []],
                    11: [0, []],
                    12: [0, []],
                    13: [0, []],
                    14: [0, []],
                    15: [0, []],
                    16: [0, []],
                    17: [0, []],
                    18: [0, []],
                    19: [0, []],
                    20: [0, []],
                    21: [0, []],
                    22: [0, []],
                    23: [0, []],
                    24: [1, [0]],
                    25: [1, [0]]
                })
                pd.testing.assert_frame_equal(exp_df, act_df)
                self.assertEqual(exp_outliers, act_outliers)

            elif num == 4:
                act_df, act_outliers = highlight_outliers(df, True)
                exp_df = pd.DataFrame({'1': [0.0001, 100],
                                       '2': [2, 2],
                                       'Analyte': ['a', 'b']})
                exp_outliers = OrderedDict({0: [0, []],
                                            1: [0, []]})
                pd.testing.assert_frame_equal(exp_df, act_df)
                self.assertEqual(exp_outliers, act_outliers)

            elif num == 5:
                act_outliers = highlight_outliers(df, False)
                exp_outliers = [['z', '1', 16.0], ['y', '1', 14.0],
                                ['b', '2', 18.0]]
                self.assertEqual(exp_outliers, act_outliers)

            elif num == 6:
                act_df, act_outliers = highlight_outliers(df, True, drop_thresh=1)
                exp_df = pd.DataFrame({
                    '1': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0],
                    '2': [3.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                          3.0, 4.0],
                    'Analyte': ['a', 'c', 'd', 'e', 'f',
                                'g', 'h', 'i', 'j', 'k', 'l',
                                'm', 'n', 'o', 'p', 'q', 'r',
                                's', 't', 'u', 'v', 'w', 'x',
                                'y', 'z']})
                exp_outliers = OrderedDict({
                    0: [0, []],
                    1: [1, [1]],
                    2: [0, []],
                    3: [0, []],
                    4: [0, []],
                    5: [0, []],
                    6: [0, []],
                    7: [0, []],
                    8: [0, []],
                    9: [0, []],
                    10: [0, []],
                    11: [0, []],
                    12: [0, []],
                    13: [0, []],
                    14: [0, []],
                    15: [0, []],
                    16: [0, []],
                    17: [0, []],
                    18: [0, []],
                    19: [0, []],
                    20: [0, []],
                    21: [0, []],
                    22: [0, []],
                    23: [0, []],
                    24: [0, []],
                    25: [0, []]
                })
                pd.testing.assert_frame_equal(exp_df, act_df)
                self.assertEqual(exp_outliers, act_outliers)

    def test_calc_median(self):
        """
        Tests calc_median in parse_array_data
        """

        print('Testing calc_median')

        plates = {1: pd.DataFrame({'1': [1.0, 2.0, 3.0],
                                   '2': [2.0, 3.0, 1.0],
                                   '3': [2.0, np.nan, 1.0]}),
                  2: pd.DataFrame({'1': [1.0, 2.0, 3.0],
                                   '2': [2.0, 3.0, 1.0],
                                   '3': [np.nan, np.nan, np.nan]})}

        for num, df in plates.items():
            if num == 1:
                act_df = calc_median(df, False)
                exp_df = pd.DataFrame({'1': [2.0],
                                       '2': [2.0],
                                       '3': [1.5]})
                pd.testing.assert_frame_equal(act_df, exp_df)
            elif num == 2:
                act_df = calc_median(df, False)
                exp_df = pd.DataFrame({'1': [2.0],
                                       '2': [2.0],
                                       '3': [np.nan]})
                pd.testing.assert_frame_equal(act_df, exp_df)

                with self.assertRaises(NaNFluorescenceError): calc_median(df, True)

    def test_scale_min_max(self):
        """
        Tests scale_min_max in parse_array_data
        """

        print('Testing scale_min_max')

        """
        plate = {'blank': pd.DataFrame({'A': [1, 2, 1],
                                        'B': [np.nan, 1, 2]}),
                 'analyte_1': pd.DataFrame({'A': [3, 4, 5],
                                            'B': [6, 3, 4]}),
                 'analyte_2': pd.DataFrame({'A': [7, 1, 2],
                                            'B': [np.nan, np.nan, 5]})}

        # Test scale method "analyte_fluorophore"
        scale_min_max(scale_method, plate, plate_name, no_pep, split_name, cols_ignore,
        plate_outliers, outlier_excl_thresh=0.05, drop_thresh=2, k_max=np.nan)

        # Test scale method "fluorophore"

        scaled_plate, plate_outliers = scale_min_max(
            scale_method, plate, plate_name, no_pep, split_name, cols_ignore,
            plate_outliers, outlier_excl_thresh=0.05, drop_thresh=2, k_max=np.nan
        )
        """

    def test_draw_boxplots(self):
        """
        Tests draw_boxplots in parse_array_data
        """

        """
        Test for np.nan and non-floats/-integers
        Test to make sure raises KeyError if analyte column isn't present
        """

        print('Testing draw_boxplots')

        exp_input_dict = {
            1: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                False, 0.2, None],
            2: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'A']],
            3: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'A']],
            4: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'A']],
            5: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'D']],
            6: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'B']],
            7: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 'X', ['B', 'C', 'A']],
            8: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None],
            9: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0]}),
                1.0, 1.0, None],
            10: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                               'Peptide 2': [3.0, 0.5, 0.6],
                               'Peptide 3': [np.nan, 1.2, 3.0],
                               'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None],
            11: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                               'Peptide 2': [3.0, np.inf, 0.6],
                               'Peptide 3': [1.7, 1.2, 3.0],
                               'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None],
            12: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                               'Peptide 2': ['X', 0.5, 0.6],
                               'Peptide 3': [1.7, 1.2, 3.0],
                               'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None]
        }
        exp_results_dict = {
            1: OrderedDict({'All_data': OrderedDict({'Original_data': pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                                                                                    'Peptide 2': [3.0, 0.5, 0.6],
                                                                                    'Peptide 3': [1.7, 1.2, 3.0],
                                                                                    'Analyte': ['A', 'C', 'B']})}),
                            'Original_data': [0.3, 4.2, pd.DataFrame({'Analyte': ['A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
                                                                      'Barrel': ['Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 3', 'Peptide 3', 'Peptide 3'],
                                                                      'Reading': [1.0, 2.0, 4.0, 3.0, 0.5, 0.6, 1.7, 1.2, 3.0]})]}),
            2: OrderedDict({'All_data': OrderedDict({'Original_data': pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                                                                                    'Peptide 2': [3.0, 0.5, 0.6],
                                                                                    'Peptide 3': [1.7, 1.2, 3.0],
                                                                                    'Analyte': ['A', 'C', 'B']}),
                                                     'Scaled_data': pd.DataFrame({'Peptide 1': [-(2/3), 0, (4/3)],
                                                                                  'Peptide 2': [1.92, -0.08, 0],
                                                                                  'Peptide 3': [0, -(5/9), (13/9)],
                                                                                  'Analyte': ['A', 'C', 'B']})}),
                            'Original_data': [-0.5, 5.0, pd.DataFrame({'Analyte': ['A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
                                                                       'Barrel': ['Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 3', 'Peptide 3', 'Peptide 3'],
                                                                       'Reading': [1.0, 2.0, 4.0, 3.0, 0.5, 0.6, 1.7, 1.2, 3.0]})],
                            'Scaled_data': [(-(2/3)-1), 2.92, pd.DataFrame({'Analyte': ['A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
                                                                            'Barrel': ['Peptide 1', 'Peptide 1', 'Peptide 1', 'Peptide 2', 'Peptide 2', 'Peptide 2', 'Peptide 3', 'Peptide 3', 'Peptide 3'],
                                                                            'Reading': [-(2/3), 0, (4/3), 1.92, -0.08, 0, 0, -(5/9), (13/9)]})]}),
            3: np.nan,
            4: np.nan,
            5: np.nan,
            6: np.nan,
            7: np.nan,
            8: np.nan,
            9: np.nan,
            10: np.nan,
            11: np.nan,
            12: np.nan
        }

        for num in exp_input_dict.keys():
            test_df = exp_input_dict[num][0]
            scale = exp_input_dict[num][1]
            cushion = exp_input_dict[num][2]
            class_order = exp_input_dict[num][3]
            exp_results = exp_results_dict[num]

            if num in [3]:
                with self.assertRaises(FileNotFoundError): draw_boxplots(
                    test_df, 'tests/Temp_output', scale, cushion, class_order, '', True
                )
            elif num in [4]:
                os.makedirs('tests/Temp_output/Boxplots')
                with self.assertRaises(FileExistsError): draw_boxplots(
                    test_df, 'tests/Temp_output', scale, cushion, class_order, '', True
                )
                shutil.rmtree('tests/Temp_output')
            elif num in [5, 6, 7, 8, 10, 11, 12]:
                os.mkdir('tests/Temp_output')
                with self.assertRaises(ValueError): draw_boxplots(
                    test_df, 'tests/Temp_output', scale, cushion, class_order, '', True
                )
                shutil.rmtree('tests/Temp_output')
            elif num in [9]:
                os.mkdir('tests/Temp_output')
                with self.assertRaises(KeyError): draw_boxplots(
                    test_df, 'tests/Temp_output', scale, cushion, class_order, '', True
                )
                shutil.rmtree('tests/Temp_output')
            else:
                os.mkdir('tests/Temp_output')
                act_results = draw_boxplots(
                    test_df, 'tests/Temp_output', scale, cushion, class_order, '', True
                )

                self.assertEqual(exp_results.keys(), act_results.keys())
                for key in exp_results.keys():
                    if key == 'All_data':
                        for sub_key in exp_results[key].keys():
                            exp_df = exp_results[key][sub_key]
                            act_df = act_results[key][sub_key]

                            pd.testing.assert_frame_equal(exp_df, act_df)
                    else:
                        exp_ymin = exp_results[key][0]
                        exp_ymax = exp_results[key][1]
                        exp_melt_df = exp_results[key][2]
                        act_ymin = act_results[key][0]
                        act_ymax = act_results[key][1]
                        act_melt_df = act_results[key][2]

                        self.assertEqual(exp_ymin, act_ymin)
                        self.assertEqual(exp_ymax, act_ymax)
                        pd.testing.assert_frame_equal(exp_melt_df, act_melt_df)

                shutil.rmtree('tests/Temp_output')

    def test_draw_heatmap_fingerprints(self):
        """
        Tests draw_heatmap_fingerprints in parse_array_data
        """

        print('Testing draw_heatmap_fingerprints')

        """
        Minor change to get tests to work after deleting and remaking deploy key
        for circleci
        """

    def test_plate_parsing(self):
        """
        Tests complete plate parsing pipeline in parse_array_data
        """

        print('Testing plate parsing pipeline')
