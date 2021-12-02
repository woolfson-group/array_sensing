
# python -m unittest tests/test_array_sensing.py

import numpy as np
import os
import pandas as pd
import pickle
import shutil
import unittest
from collections import OrderedDict
from subroutines.exceptions import (
    MinMaxFluorescenceError, NaNFluorescenceError, PlateLayoutError
)
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
                                      '2': [np.nan, np.nan, np.nan],
                                      'Analyte': ['a', 'b', 'c']}),
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
                                                  'y', 'z']}),
                     7: pd.DataFrame({'1': [0.0001, 100],
                                      '2': [2, 2]})}

        for num, df in input_dfs.items():
            if num in [1, 2, 7]:
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

        exp_input_dict = {
            # Test function scaling with "fluorophore" method
            1: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [0.3, 0.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'fluorophore', 'Peptide 1', 'Split 1', [], [], 0.05, 1, 3, False],
            # Test function scaling with "analyte_fluorophore" method
            2: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [0.3, 0.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'analyte_fluorophore', 'Peptide 1', 'Split 1', [], [], 0.05, 1, 3, False],
            # Test cols_ignore
            3: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [0.3, 0.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'fluorophore', 'Peptide 1', 'Split 1', ['Split 1_Peptide 3', 'Split 1_Peptide 4'], [], 0.05, 1, 3, False],
            # Test detection of peptide + DPH < peptide + analyte + DPH
            4: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [0.3, 0.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'fluorophore', 'Peptide 1', 'Split 1', [], [], 0.05, 1, 3, True],
            # Test min max scaling divide by zero error
            5: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [1.3, 5.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'analyte_fluorophore', 'Peptide 1', 'Split 1', [], [], 0.05, 1, 3, False],
            # Test min max scaling divide by negative number error
            6: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [5.9, 5.1, 6.2],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'fluorophore', 'Peptide 1', 'Split 1', [], [], 0.05, 1, 3, False],
            # Test requirement for "blank" entry in input dictionary
            7: [OrderedDict({'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'fluorophore', 'Peptide 1', 'Split 1', [], [], 0.05, 1, 3, False],
            # Test "No pep" not in dataframe
            8: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [0.3, 0.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'fluorophore', 'Peptide 4', 'Split 1', [], [], 0.05, 1, 3, False],
            # Test outlier detection within scale_min_max function (no need to
            # test multiple values for outlier_excl_threshold, drop_thresh or
            # k_max, as these have already been tested for the highlight_outliers
            # function that is called by scale_min_max)
            9: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [0.3, 0.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.5],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'fluorophore', 'Peptide 1', 'Split 1', [], [], 0.1, 1, 3, False],
            # Test NaN value detection
            10: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [0.3, 0.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [np.nan, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'fluorophore', 'Peptide 1', 'Split 1', [], [], 0.05, 1, 3, False],
            # Test scale_method value detection
            11: [OrderedDict({'blank': pd.DataFrame({'Split 1_Peptide 1': [0.3, 0.3, 180],
                                                    'Split 1_Peptide 2': [3.3, 3.0, 3.4],
                                                    'Split 1_Peptide 3': [9.3, 9.2, 12.0],
                                                    'Analyte': ['blank', 'blank', 'blank'],
                                                    'Plate': ['', '', '']}),
                             'Analyte 2': pd.DataFrame({'Split 1_Peptide 1': [0.4, 0.6, 0.1],
                                                        'Split 1_Peptide 2': [1.8, 2.3, 2.6],
                                                        'Split 1_Peptide 3': [3.5, 6.0, 4.2],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 1': [2.3, 2.9, 3.0],
                                                        'Split 1_Peptide 2': [8.1, 0.1, 5.6],
                                                        'Split 1_Peptide 3': [2.12, 2.1, 2.2],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                'X', 'Peptide 1', 'Split 1', [], [], 0.05, 1, 3, False]
        }

        exp_results_dict = {
            1: [OrderedDict({'Analyte 2': pd.DataFrame({'Split 1_Peptide 2': [(1.5/3), (2/3), (2.3/3)],
                                                        'Split 1_Peptide 3': [(3.2/9), (5.7/9), (3.9/9)],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 2': [(7.8/3), (-0.2/3), (5.3/3)],
                                                        'Split 1_Peptide 3': [(1.82/9), (1.8/9), (1.9/9)],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                [['blank', 'Split 1_Peptide 1', 180]]],
            2: [OrderedDict({'Analyte 2': pd.DataFrame({'Split 1_Peptide 2': [(1.4/3), (1.9/3), (2.2/3)],
                                                        'Split 1_Peptide 3': [(3.1/9), (5.6/9), (3.8/9)],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 2': [(5.2/3), (-2.8/3), (2.7/3)],
                                                        'Split 1_Peptide 3': [(-0.78/9), (-0.8/9), (-0.7/9)],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                [['blank', 'Split 1_Peptide 1', 180]]],
            3: [OrderedDict({'Analyte 2': pd.DataFrame({'Split 1_Peptide 2': [(1.5/3), (2/3), (2.3/3)],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 2': [(7.8/3), (-0.2/3), (5.3/3)],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                [['blank', 'Split 1_Peptide 1', 180]]],
            4: [np.nan, np.nan],
            5: [np.nan, np.nan],
            6: [np.nan, np.nan],
            7: [np.nan, np.nan],
            8: [np.nan, np.nan],
            9: [OrderedDict({'Analyte 2': pd.DataFrame({'Split 1_Peptide 2': [(1.5/3), (2/3), (2.3/3)],
                                                        'Split 1_Peptide 3': [(3.2/9), (5.7/9), (3.9/9)],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 2': [(7.8/3), (-0.2/3), (5.3/3)],
                                                        'Split 1_Peptide 3': [(1.82/9), (1.8/9), (2.2/9)],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                [['blank', 'Split 1_Peptide 1', 180], ['blank', 'Split 1_Peptide 3', 12], ['Analyte 1', 'Split 1_Peptide 3', 2.5]]],
            10: [np.nan, np.nan],
            11: [np.nan, np.nan]
        }

        for num in exp_input_dict.keys():
            plate = exp_input_dict[num][0]
            scale_method = exp_input_dict[num][1]
            no_pep = exp_input_dict[num][2]
            split_name = exp_input_dict[num][3]
            cols_ignore = exp_input_dict[num][4]
            plate_outliers = exp_input_dict[num][5]
            alpha_generalised_esd = exp_input_dict[num][6]
            drop_thresh = exp_input_dict[num][7]
            k_max = exp_input_dict[num][8]
            test = exp_input_dict[num][9]
            exp_scaled_plate = exp_results_dict[num][0]
            exp_plate_outliers = exp_results_dict[num][1]

            if num in [4, 10, 11]:
                with self.assertRaises(ValueError): scale_min_max(
                    scale_method, plate, '', no_pep, split_name,
                    cols_ignore, plate_outliers, alpha_generalised_esd,
                    drop_thresh, k_max, test
                )
            elif num in [5, 6]:
                with self.assertRaises(MinMaxFluorescenceError): scale_min_max(
                    scale_method, plate, '', no_pep, split_name,
                    cols_ignore, plate_outliers, alpha_generalised_esd,
                    drop_thresh, k_max, test
                )
            elif num in [7, 8]:
                with self.assertRaises(PlateLayoutError): scale_min_max(
                    scale_method, plate, '', no_pep, split_name,
                    cols_ignore, plate_outliers, alpha_generalised_esd,
                    drop_thresh, k_max, test
                )
            else:
                act_scaled_plate, act_plate_outliers = scale_min_max(
                    scale_method, plate, '', no_pep, split_name,
                    cols_ignore, plate_outliers, alpha_generalised_esd,
                    drop_thresh, k_max, test
                )
                self.assertEqual(list(exp_scaled_plate.keys()), list(act_scaled_plate.keys()))
                for key in exp_scaled_plate.keys():
                    exp_df = exp_scaled_plate[key]
                    act_df = act_scaled_plate[key]
                    pd.testing.assert_frame_equal(exp_df, act_df)
                self.assertEqual(exp_plate_outliers, act_plate_outliers)

    def test_draw_boxplots(self):
        """
        Tests draw_boxplots in parse_array_data
        """

        print('Testing draw_boxplots')

        exp_input_dict = {
            # Test function (without scaling)
            1: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                False, 0.2, None],
            # Test function (with scaling)
            2: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'A']],
            # Test directory exists errors
            3: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'A']],
            # Test directory exists errors
            4: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'A']],
            # Test for detection of unrecognised analytes
            5: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'D']],
            # Test for detection of duplicate analytes
            6: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 1.0, ['B', 'C', 'B']],
            # Test for detection of non-numeric value for cushion
            7: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, 'X', ['B', 'C', 'A']],
            # Test for detection of non-Boolean value for scale
            8: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None],
            # Test for detecting "Analyte" column in dataframe
            9: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0]}),
                1.0, 1.0, None],
            # Test for detecting NaN values in dataframe
            10: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                               'Peptide 2': [3.0, 0.5, 0.6],
                               'Peptide 3': [np.nan, 1.2, 3.0],
                               'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None],
            # Test for detecting infinite values in dataframe
            11: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                               'Peptide 2': [3.0, np.inf, 0.6],
                               'Peptide 3': [1.7, 1.2, 3.0],
                               'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None],
            # Test for detecting non-numeric values in dataframe
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

                self.assertEqual(list(exp_results.keys()), list(act_results.keys()))
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

                        np.testing.assert_almost_equal(exp_ymin, act_ymin, 7)
                        np.testing.assert_almost_equal(exp_ymax, act_ymax, 7)
                        pd.testing.assert_frame_equal(exp_melt_df, act_melt_df)

                shutil.rmtree('tests/Temp_output')

    def test_draw_heatmap_fingerprints(self):
        """
        Tests draw_heatmap_fingerprints in parse_array_data
        """

        print('Testing draw_heatmap_fingerprints')

        exp_input_dict = {
            # Test function (without scaling)
            1: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'A']}),
                False, ['C', 'A']],
            # Test function (with scaling)
            2: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['B', 'B', 'A']}),
                True, None],
            # Test directory exists errors
            3: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, ['B', 'C', 'A']],
            # Test directory exists errors
            4: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, ['B', 'C', 'A']],
            # Test for detection of unrecognised analytes
            5: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, ['B', 'C', 'D']],
            # Test for detection of duplicate analytes
            6: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                True, ['B', 'C', 'B']],
            # Test for detection of non-Boolean value for scale
            7: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                1.0, ['B', 'C', 'A']],
            # Test for detecting "Analyte" column in dataframe
            8: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [1.7, 1.2, 3.0]}),
                1.0, 1.0, None],
            # Test for detecting NaN values in dataframe
            9: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                              'Peptide 2': [3.0, 0.5, 0.6],
                              'Peptide 3': [np.nan, 1.2, 3.0],
                              'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None],
            # Test for detecting infinite values in dataframe
            10: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                               'Peptide 2': [3.0, np.inf, 0.6],
                               'Peptide 3': [1.7, 1.2, 3.0],
                               'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None],
            # Test for detecting non-numeric values in dataframe
            11: [pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                               'Peptide 2': ['X', 0.5, 0.6],
                               'Peptide 3': [1.7, 1.2, 3.0],
                               'Analyte': ['A', 'C', 'B']}),
                1.0, 1.0, None]
        }

        exp_results_dict = {
            1: OrderedDict({'All_data': OrderedDict({'Original_data': pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                                                                                    'Peptide 2': [3.0, 0.5, 0.6],
                                                                                    'Peptide 3': [1.7, 1.2, 3.0],
                                                                                    'Analyte': ['A', 'C', 'A']})}),
                            'Original_data': OrderedDict({'min': 0.5,
                                                          'max': 2.5,
                                                          'C': np.array([2.0, 0.5, 1.2]),
                                                          'A': np.array([2.5, 1.8, 2.35])})}),
            2: OrderedDict({'All_data': OrderedDict({'Original_data': pd.DataFrame({'Peptide 1': [1.0, 2.0, 4.0],
                                                                                    'Peptide 2': [3.0, 0.5, 0.6],
                                                                                    'Peptide 3': [1.7, 1.2, 3.0],
                                                                                    'Analyte': ['B', 'B', 'A']}),
                                                     'Scaled_data': pd.DataFrame({'Peptide 1': [-(2/3), 0, (4/3)],
                                                                                  'Peptide 2': [1.92, -0.08, 0],
                                                                                  'Peptide 3': [0, -(5/9), (13/9)],
                                                                                  'Analyte': ['B', 'B', 'A']})}),
                            'Original_data': OrderedDict({'min': 0.6,
                                                          'max': 4.0,
                                                          'A': np.array([4.0, 0.6, 3.0]),
                                                          'B': np.array([1.5, 1.75, 1.45])}),
                            'Scaled_data': OrderedDict({'min': -(1/3),
                                                        'max': (13/9),
                                                        'A': np.array([(4/3), 0, (13/9)]),
                                                        'B': np.array([-(1/3), 0.92, -(5/18)])})}),
            3: np.nan,
            4: np.nan,
            5: np.nan,
            6: np.nan,
            7: np.nan,
            8: np.nan,
            9: np.nan,
            10: np.nan,
            11: np.nan
        }

        for num in exp_input_dict.keys():
            test_df = exp_input_dict[num][0]
            scale = exp_input_dict[num][1]
            class_order = exp_input_dict[num][2]
            exp_results = exp_results_dict[num]

            if num in [3]:
                with self.assertRaises(FileNotFoundError): draw_heatmap_fingerprints(
                    test_df, 'tests/Temp_output', scale, class_order, '', True
                )
            elif num in [4]:
                os.makedirs('tests/Temp_output/Heatmap_plots')
                with self.assertRaises(FileExistsError): draw_heatmap_fingerprints(
                    test_df, 'tests/Temp_output', scale, class_order, '', True
                )
                shutil.rmtree('tests/Temp_output')
            elif num in [5, 6, 7, 9, 10, 11]:
                os.mkdir('tests/Temp_output')
                with self.assertRaises(ValueError): draw_heatmap_fingerprints(
                    test_df, 'tests/Temp_output', scale, class_order, '', True
                )
                shutil.rmtree('tests/Temp_output')
            elif num in [8]:
                os.mkdir('tests/Temp_output')
                with self.assertRaises(KeyError): draw_heatmap_fingerprints(
                    test_df, 'tests/Temp_output', scale, class_order, '', True
                )
                shutil.rmtree('tests/Temp_output')
            else:
                os.mkdir('tests/Temp_output')

                act_results = draw_heatmap_fingerprints(
                    test_df, 'tests/Temp_output', scale, class_order, '', True
                )

                self.assertEqual(list(exp_results.keys()), list(act_results.keys()))
                for key in exp_results.keys():
                    for sub_key in exp_results[key].keys():
                        if sub_key in ['min', 'max']:
                            exp_min_max = exp_results[key][sub_key]
                            act_min_max = act_results[key][sub_key]
                            np.testing.assert_almost_equal(exp_min_max, act_min_max, 7)
                        else:
                            exp_array = exp_results[key][sub_key]
                            act_array = act_results[key][sub_key]
                            if key == 'All_data':
                                pd.testing.assert_frame_equal(exp_array, act_array)
                            else:
                                np.testing.assert_almost_equal(exp_array, act_array, 7)

                shutil.rmtree('tests/Temp_output')


    def test_plate_parsing(self):
        """
        Tests complete plate parsing pipeline in parse_array_data
        """

        print('Testing plate parsing pipeline')
