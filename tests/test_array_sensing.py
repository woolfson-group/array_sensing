
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
    draw_boxplots, draw_heatmap_fingerprints, ParseArrayData
)


class TestClass(unittest.TestCase):

    def test_trim_dataframe(self):
        """
        Tests trim_dataframe in parse_array_data
        """

        print('Testing trim_dataframe')

        exp_input_dict = {
            1: pd.DataFrame(np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]])),
            2: pd.DataFrame(np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [np.nan, np.nan, np.nan]])),
            3: pd.DataFrame(np.array([[1.0, 2.0, np.nan],
                                      [4.0, 5.0, np.nan],
                                      [7.0, 8.0, np.nan]])),
            4: pd.DataFrame(np.array([[1.0, np.nan, 3.0],
                                      [4.0, np.nan, 6.0],
                                      [7.0, np.nan, 9.0]])),
            5: pd.DataFrame(np.array([[1.0, 2.0, 3.0],
                                      [np.nan, np.nan, np.nan],
                                      [7.0, 8.0, 9.0]])),
            6: pd.DataFrame(np.array([[1.0, np.nan, 3.0],
                                      [np.nan, np.nan, 6.0],
                                      [7.0, 8.0, 9.0]])),
            7: pd.DataFrame(np.array([['1', 2.0, np.nan],
                                      [np.nan, np.nan, np.nan],
                                      [7.0, 8.0, '9']], dtype=object)),
            8: pd.DataFrame({}),
            9: pd.DataFrame(np.array([[np.nan, np.nan, np.nan],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]])),
            10: pd.DataFrame(np.array([[np.nan, 2.0, 3.0],
                                       [np.nan, 5.0, 6.0],
                                       [np.nan, 8.0, 9.0]]))
        }
        exp_results_dict = {
            1: pd.DataFrame(np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]])),
            2: pd.DataFrame(np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])),
            3: pd.DataFrame(np.array([[1.0, 2.0],
                                      [4.0, 5.0],
                                      [7.0, 8.0]])),
            4: pd.DataFrame(np.array([[1.0],
                                      [4.0],
                                      [7.0]])),
            5: pd.DataFrame(np.array([[1.0, 2.0, 3.0]])),
            6: pd.DataFrame(np.array([[1.0, np.nan, 3.0],
                                      [np.nan, np.nan, 6.0],
                                      [7.0, 8.0, 9.0]])),
            7: pd.DataFrame(np.array([['1', 2.0]], dtype=object)),
            8: np.nan,
            9: np.nan,
            10: np.nan}

        for num in exp_input_dict.keys():
            input_df = exp_input_dict[num]
            exp_df = exp_results_dict[num]

            if num in range(1, 8):
                act_df = trim_dataframe(input_df, '')
                pd.testing.assert_frame_equal(exp_df, act_df)
            elif num in range(8, 11):
                with self.assertRaises(PlateLayoutError): trim_dataframe(input_df, '')

    def test_parse_xlsx_to_dataframe(self):
        """
        Tests parse_xlsx_to_dataframe in parse_array_data
        """

        print('Testing parse_xlsx_to_dataframe')

        exp_input_dict = {
            # Test plate parsing
            1: ['tests/Test_plates/Test_plate_1.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test non-numeric value for gain
            2: ['tests/Test_plates/Test_plate_2.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 'X'],
            # Test non-integer value for gain
            3: ['tests/Test_plates/Test_plate_3.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 4.2],
            # Test value less than or equal to 0 for gain
            4: ['tests/Test_plates/Test_plate_4.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 0],
            # Test plate path that doesn't exist
            5: ['tests/Test_plates/Test_plate_5.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test no "Protocol Information" sheet
            6: ['tests/Test_plates/Test_plate_6.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test no "platelayout" on "Protocol Information" sheet
            7: ['tests/Test_plates/Test_plate_7.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test no "peptidelayout" on "Protocol Information" sheet
            8: ['tests/Test_plates/Test_plate_8.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test no "End point" sheet
            9: ['tests/Test_plates/Test_plate_9.xlsx', 'Split 1',
                {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                 'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test no "{}.rawdata" marking the start of the plate data on "End
            # point" sheet
            10: ['tests/Test_plates/Test_plate_10.xlsx', 'Split 1',
                 {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                  'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test peptides supplied by the user don't match those on the plate
            11: ['tests/Test_plates/Test_plate_11.xlsx', 'Split 1',
                 {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 5'],
                  'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test peptides supplied by the user don't match those on the plate
            12: ['tests/Test_plates/Test_plate_12.xlsx', 'Split 1',
                 {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1'],
                  'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test peptides supplied by the user don't contain repeated values
            13: ['tests/Test_plates/Test_plate_13.xlsx', 'Split 1',
                 {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3', 'Peptide 3'],
                  'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test peptides aren't listed more than once in platelayout
            14: ['tests/Test_plates/Test_plate_14.xlsx', 'Split 1',
                 {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                  'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test non-standard shape of peptide layout
            15: ['tests/Test_plates/Test_plate_15.xlsx', 'Split 1',
                 {'Split 1': ['Peptide 2', 'Peptide 1', 'Peptide 3'],
                  'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Test non-standard plate layout
            16: ['tests/Test_plates/Test_plate_16.xlsx', 'Split 1',
                 {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                  'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1],
            # Tests detection of NaN values
            17: ['tests/Test_plates/Test_plate_17.xlsx', 'Split 1',
                 {'Split 1': ['Peptide 4', 'Peptide 2', 'Peptide 1', 'Peptide 3'],
                  'Split 2': ['Peptide 1', 'Peptide 4', 'Peptide 2', 'Peptide 5']}, 1]
        }

        exp_results_dict = {
            1: [pd.DataFrame({'A': [0.469, 9.85, 1.12, 8.21, 1, 0.379],
                              'B': [1.72, 7.66, 5.68, 9.81, 9.19, 4.5],
                              'C': [9.81, 9.3, 3.42, 1.04, 7.46, 3.6],
                              'D': [5.42, 3.89, 3.44, 0.448, 4.97, 6.94],
                              'E': [8.1, 8.28, 4.37, 5.03, 3.4, 0.517],
                              'F': [2.75, 4.32, 9.75, 2.44, 8.33, 9.22]}),
                OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_1.xlsx' for n in range(6)],
                                                        'Analyte': ['Analyte 1' for n in range(6)],
                                                        'Split 1_Peptide 4': [7.66, 3.89, 2.44, 4.5, 6.94, 9.22],
                                                        'Split 1_Peptide 2': [9.85, 9.3, 5.03, 0.379, 3.6, 0.517],
                                                        'Split 1_Peptide 1': [0.469, 9.81, 4.37, 1.0, 7.46, 3.4],
                                                        'Split 1_Peptide 3': [1.72, 5.42, 9.75, 9.19, 4.97, 8.33]}, dtype=object),
                             'Analyte X': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_1.xlsx'],
                                                        'Analyte': ['Analyte X'],
                                                        'Split 1_Peptide 4': [4.32],
                                                        'Split 1_Peptide 2': [8.28],
                                                        'Split 1_Peptide 1': [8.1],
                                                        'Split 1_Peptide 3': [2.75]}, dtype=object),
                             'Analyte 2': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_1.xlsx' for n in range(2)],
                                                        'Analyte': ['Analyte 2' for n in range(2)],
                                                        'Split 1_Peptide 4': [9.81, 0.448],
                                                        'Split 1_Peptide 2': [8.21, 1.04],
                                                        'Split 1_Peptide 1': [1.12, 3.42],
                                                        'Split 1_Peptide 3': [5.68, 3.44]}, dtype=object)}),
                ['Split 1_Peptide 4', 'Split 1_Peptide 2', 'Split 1_Peptide 1',
                 'Split 1_Peptide 3']],
            2: [np.nan, np.nan, np.nan],
            3: [np.nan, np.nan, np.nan],
            4: [np.nan, np.nan, np.nan],
            5: [np.nan, np.nan, np.nan],
            6: [np.nan, np.nan, np.nan],
            7: [np.nan, np.nan, np.nan],
            8: [np.nan, np.nan, np.nan],
            9: [np.nan, np.nan, np.nan],
            10: [np.nan, np.nan, np.nan],
            11: [np.nan, np.nan, np.nan],
            12: [np.nan, np.nan, np.nan],
            13: [np.nan, np.nan, np.nan],
            14: [np.nan, np.nan, np.nan],
            15: [pd.DataFrame({'A': [0.469, 9.85, 1.12, 8.21, 1, 0.379],
                               'B': [1.72, np.nan, 5.68, 9.81, 9.19, 4.5],
                               'C': [9.81, 9.3, 3.42, 1.04, 7.46, 3.6],
                               'D': [5.42, 3.89, 3.44, 0.448, 4.97, 6.94],
                               'E': [8.1, 8.28, 4.37, 5.03, 3.4, 0.517],
                               'F': [2.75, 4.32, 9.75, 2.44, 8.33, 9.22]}),
                    OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_15.xlsx' for n in range(6)],
                                                            'Analyte': ['Analyte 1' for n in range(6)],
                                                            'Split 1_Peptide 2': [9.85, 9.3, 5.03, 0.379, 3.6, 0.517],
                                                            'Split 1_Peptide 1': [0.469, 9.81, 4.37, 1.0, 7.46, 3.4],
                                                            'Split 1_Peptide 3': [1.72, 5.42, 9.75, 9.19, 4.97, 8.33]}, dtype=object),
                                 'Analyte X': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_15.xlsx'],
                                                            'Analyte': ['Analyte X'],
                                                            'Split 1_Peptide 2': [8.28],
                                                            'Split 1_Peptide 1': [8.1],
                                                            'Split 1_Peptide 3': [2.75]}, dtype=object),
                                 'Analyte 2': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_15.xlsx' for n in range(2)],
                                                            'Analyte': ['Analyte 2' for n in range(2)],
                                                            'Split 1_Peptide 2': [8.21, 1.04],
                                                            'Split 1_Peptide 1': [1.12, 3.42],
                                                            'Split 1_Peptide 3': [5.68, 3.44]}, dtype=object)}),
                    ['Split 1_Peptide 2', 'Split 1_Peptide 1', 'Split 1_Peptide 3']],
            16: [pd.DataFrame({'A': [0.469, 9.85, 1.12, 8.21, 1, 0.379],
                               'B': [1.72, 7.66, 5.68, 9.81, 9.19, 4.5],
                               'C': [9.81, 9.3, 3.42, 1.04, 7.46, 3.6],
                               'D': [5.42, 3.89, 3.44, 0.448, 4.97, 6.94],
                               'E': [8.1, 8.28, 4.37, 5.03, 3.4, 0.517],
                               'F': [2.75, 4.32, 9.75, 2.44, 8.33, 9.22]}),
                 OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_16.xlsx' for n in range(4)],
                                                         'Analyte': ['Analyte 1' for n in range(4)],
                                                         'Split 1_Peptide 4': [3.89, 2.44, 4.5, 9.22],
                                                         'Split 1_Peptide 2': [9.3, 5.03, 0.379, 0.517],
                                                         'Split 1_Peptide 1': [9.81, 4.37, 1.0, 3.4],
                                                         'Split 1_Peptide 3': [5.42, 9.75, 9.19, 8.33]}, dtype=object),
                              'Analyte X': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_16.xlsx'],
                                                         'Analyte': ['Analyte X'],
                                                         'Split 1_Peptide 4': [4.32],
                                                         'Split 1_Peptide 2': [8.28],
                                                         'Split 1_Peptide 1': [8.1],
                                                         'Split 1_Peptide 3': [2.75]}, dtype=object),
                              'Analyte 2': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_16.xlsx' for n in range(2)],
                                                         'Analyte': ['Analyte 2' for n in range(2)],
                                                         'Split 1_Peptide 4': [9.81, 0.448],
                                                         'Split 1_Peptide 2': [8.21, 1.04],
                                                         'Split 1_Peptide 1': [1.12, 3.42],
                                                         'Split 1_Peptide 3': [5.68, 3.44]}, dtype=object)}),
                 ['Split 1_Peptide 4', 'Split 1_Peptide 2', 'Split 1_Peptide 1',
                  'Split 1_Peptide 3']],
            17: [np.nan, np.nan, np.nan]
        }

        for num in exp_input_dict.keys():
            plate_path = exp_input_dict[num][0]
            split_name = exp_input_dict[num][1]
            peptide_dict = exp_input_dict[num][2]
            gain = exp_input_dict[num][3]
            exp_raw_data = exp_results_dict[num][0]
            exp_grouped_data = exp_results_dict[num][1]
            exp_peptide_list = exp_results_dict[num][2]

            if num in [2, 3]:
                with self.assertRaises(TypeError): parse_xlsx_to_dataframe(
                    plate_path, split_name, peptide_dict, gain
                )
            elif num in [5]:
                with self.assertRaises(FileNotFoundError): parse_xlsx_to_dataframe(
                    plate_path, split_name, peptide_dict, gain
                )
            elif num in [4, 6, 7, 8, 9, 10, 17]:
                with self.assertRaises(ValueError): parse_xlsx_to_dataframe(
                    plate_path, split_name, peptide_dict, gain
                )
            elif num in [11, 12, 13, 14]:
                with self.assertRaises(PlateLayoutError): parse_xlsx_to_dataframe(
                    plate_path, split_name, peptide_dict, gain=1
                )
            else:
                (
                    act_raw_data, act_grouped_data, act_peptide_list
                ) = parse_xlsx_to_dataframe(
                    plate_path, split_name, peptide_dict, gain
                )

                pd.testing.assert_frame_equal(exp_raw_data, act_raw_data)
                self.assertEqual(list(exp_grouped_data.keys()), list(act_grouped_data.keys()))
                for key in exp_grouped_data.keys():
                    exp_df = exp_grouped_data[key]
                    act_df = act_grouped_data[key]
                    pd.testing.assert_frame_equal(exp_df, act_df)
                self.assertEqual(exp_peptide_list, act_peptide_list)

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
                'fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, False],
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
                'analyte_fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, False],
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
                'fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', ['Split 1_Peptide 3', 'Split 1_Peptide 4'], [], 0.05, 1, 3, False],
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
                'fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, True],
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
                'analyte_fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, False],
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
                'fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, False],
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
                'fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, False],
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
                'fluorophore', {'Split 1': 'Peptide 4', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, False],
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
                'fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.1, 1, 3, False],
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
                'fluorophore', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, False],
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
                'X', {'Split 1': 'Peptide 1', 'Split N': 'Peptide N'}, 'Split 1', [], [], 0.05, 1, 3, False]
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
                [['blank', 'Split 1_Peptide 1', 180]],
                ['Split 1_Peptide 2', 'Split 1_Peptide 3']],
            2: [OrderedDict({'Analyte 2': pd.DataFrame({'Split 1_Peptide 2': [(1.4/3), (1.9/3), (2.2/3)],
                                                        'Split 1_Peptide 3': [(3.1/9), (5.6/9), (3.8/9)],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 2': [(5.2/3), (-2.8/3), (2.7/3)],
                                                        'Split 1_Peptide 3': [(-0.78/9), (-0.8/9), (-0.7/9)],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                [['blank', 'Split 1_Peptide 1', 180]],
                ['Split 1_Peptide 2', 'Split 1_Peptide 3']],
            3: [OrderedDict({'Analyte 2': pd.DataFrame({'Split 1_Peptide 2': [(1.5/3), (2/3), (2.3/3)],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 2': [(7.8/3), (-0.2/3), (5.3/3)],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                [['blank', 'Split 1_Peptide 1', 180]],
                ['Split 1_Peptide 2']],
            4: [np.nan, np.nan, np.nan],
            5: [np.nan, np.nan, np.nan],
            6: [np.nan, np.nan, np.nan],
            7: [np.nan, np.nan, np.nan],
            8: [np.nan, np.nan, np.nan],
            9: [OrderedDict({'Analyte 2': pd.DataFrame({'Split 1_Peptide 2': [(1.5/3), (2/3), (2.3/3)],
                                                        'Split 1_Peptide 3': [(3.2/9), (5.7/9), (3.9/9)],
                                                        'Analyte': ['Analyte 2', 'Analyte 2', 'Analyte 2'],
                                                        'Plate': ['', '', '']}),
                             'Analyte 1': pd.DataFrame({'Split 1_Peptide 2': [(7.8/3), (-0.2/3), (5.3/3)],
                                                        'Split 1_Peptide 3': [(1.82/9), (1.8/9), (2.2/9)],
                                                        'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 1'],
                                                        'Plate': ['', '', '']})}),
                [['blank', 'Split 1_Peptide 1', 180], ['blank', 'Split 1_Peptide 3', 12], ['Analyte 1', 'Split 1_Peptide 3', 2.5]],
                ['Split 1_Peptide 2', 'Split 1_Peptide 3']],
            10: [np.nan, np.nan, np.nan],
            11: [np.nan, np.nan, np.nan]
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
            exp_peptide_list = exp_results_dict[num][2]

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
                (
                    act_scaled_plate, act_plate_outliers, act_peptide_list
                ) = scale_min_max(
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
                self.assertEqual(exp_peptide_list, act_peptide_list)

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

        for n in range(2):
            if n == 0:
                # Expected values
                exp_peptide_list = [
                    'Split 1_Peptide 2', 'Split 1_Peptide X', 'Split 1_Peptide 4',
                    'Split 2_Peptide 1', 'Split 2_Peptide 2'
                ]
                exp_repeat_dict = OrderedDict({
                    '_001': OrderedDict({'Split 1': ['Test_plate_A_001.xlsx',
                                                     'Test_plate_B_001.xlsx'],
                                         'Split 2': ['Test_plate_X_001.xlsx',
                                                     'Test_plate_Y_001.xlsx']}),
                    '_002': OrderedDict({'Split 1': ['Test_plate_B_002.xlsx'],
                                         'Split 2': ['Test_plate_K_002.xlsx',
                                                     'Test_plate_L_002.xlsx']})})
                exp_plates = OrderedDict({
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_A_001.xlsx': pd.DataFrame({
                        'A': [0.469, 9.85, 1.12, 8.21, 1, 0.379],
                        'B': [13.9, 7.66, 12.2, 9.81, 9.49, 4.5],
                        'C': [9.81, 9.3, 3.42, 1.04, 7.46, 3.6],
                        'D': [9.42, 3.89, 7.45, 0.448, 14, 6.94],
                        'E': [8.1, 8.28, 4.37, 5.03, 3.4, 9.22],
                        'F': [10.6, 4.32, 14.3, 2.44, 8.17, 0.517]
                    }),
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_001.xlsx': pd.DataFrame({
                        'A': [9.63, 2.23, 3.46, 6.76, 8.15, 14.91],
                        'B': [9.63, 3.71, 8.56, 7.06, 7.9, 8.36],
                        'C': [np.nan, 4.44, 6.31, 6.31, 2.35, 1.15],
                        'D': [12.2, 8.86, 6.84, 6.57, 13.8, 7.81],
                        'E': [8.49, 6.99, 9.5, 1.14, 0.304, 8.02],
                        'F': [8.19, 8.29, 8.26, 6.84, 9.32, 9.77]
                    }),
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_X_001.xlsx': pd.DataFrame({
                        'A': [2.05, 6.12, 0.0316, 7.74, 5.24, 4.51],
                        'B': [14.9, 7.85, 11.7, 4.06, 12.8, 8.85],
                        'C': [0.291, 1.21, 4.22, 2.18, 0.618, 4.05],
                        'D': [10.4, 3.78, 13, 1.78, 7.38, 9.88],
                        'E': [1.23, 5.07, 5.1, 13, 4.62, 9.81],
                        'F': [5.97, 2.06, 7.01, 3.83, 9.04, 6.06]
                    }),
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_Y_001.xlsx': pd.DataFrame({
                        'A': [1.66, 1.75, 0.654, 8.24, 7.24, 6.19],
                        'B': [2.25, 5.35, 3.51, 11.4, 4.68, 12],
                        'C': [0.778, 2.94, 8.06, 9.26, 3.43, 3.83],
                        'D': [7.73, 11.4, 0.312, 12, 6.71, 14],
                        'E': [7.05, 7.71, 4.9, 6.98, 1.09, 4.74],
                        'F': [9.16, 11.7, 5.15, 5.49, 0.868, 13.3]
                    }),
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_002.xlsx': pd.DataFrame({
                        'A': [3.07, 0.0732, 5.8, 5.58, 4.54, 9.19],
                        'B': [7.07, 7.93, 13.2, 0.568, 6.48, 7.46],
                        'C': [6.45, 1.42, 6.24, 3.26, 6.02, 0.786],
                        'D': [14.4, 0.465, 6.1, 5.09, 9, 2.19],
                        'E': [5.79, 6.95, 6.65, 6.96, 2.64, 9.47],
                        'F': [5.3, 3.29, 10.5, 3.03, 13.8, 7.01]
                    }),
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_K_002.xlsx': pd.DataFrame({
                        'A': [6.43, 5, 8.46, 7.44, 1.32, 6.25],
                        'B': [9.66, 8.11, 9.5, 0.502, 8, 6.37],
                        'C': [6.19, 3.88, 7.39, 9.92, 6.28, 5.56],
                        'D': [9.48, 8.17, 2.92, 5.82, 3.62, 0.814],
                        'E': [11.9, 9.48, 9.5, 5.42, 13.5, 1.88],
                        'F': [4, 7.93, 5.94, 6.09, 8.39, 1.18]
                    }),
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_L_002.xlsx': pd.DataFrame({
                        'A': [2.21, 10.8, 2.31, 7.88, 1.45, 12.3],
                        'B': [2.02, 8.61, 8.55, 12.8, 6.54, 14.5],
                        'C': [6.14, 5.08, 1.51, 8, 11.2, 7.41]
                    })
                })
                exp_scaled_data = OrderedDict({
                    '_001': OrderedDict({'Split 1': OrderedDict({
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_A_001.xlsx': OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_A_001.xlsx' for n in range(5)], 'Analyte': ['Analyte 1' for n in range(5)], 'Split 1_Peptide 2': [0.614730553, 0.551533954, 0.060898541, -0.47351488, -0.103412616]}, dtype=object),
                                                                                                                    'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_A_001.xlsx'], 'Analyte': ['Analyte 3'], 'Split 1_Peptide 2': [0.455015512]}, dtype=object)}),
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_001.xlsx': OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_001.xlsx' for n in range(5)], 'Analyte': ['Analyte 1' for n in range(5)], 'Split 1_Peptide 2': [-0.851908397, -0.514503817, -1.018320611, -1.016793893, 0.032061069]}, dtype=object),
                                                                                                                    'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_001.xlsx'], 'Analyte': ['Analyte 3'], 'Split 1_Peptide 2': [-0.198473282]}, dtype=object)})
                    }),
                                         'Split 2': OrderedDict({
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_X_001.xlsx': OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_X_001.xlsx' for n in range(5)], 'Analyte': ['Analyte 1' for n in range(5)], 'Split 2_Peptide 2': [-0.714524207, -1.534223706, -0.983305509, -1.060100167, -0.098497496]}, dtype=object),
                                                                                                                    'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_X_001.xlsx'], 'Analyte': ['Analyte 3'], 'Split 2_Peptide 2': [-0.150250417]}, dtype=object)}),
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_Y_001.xlsx': OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_Y_001.xlsx' for n in range(5)], 'Analyte': ['Analyte 1' for n in range(5)], 'Split 2_Peptide 2': [-0.619450317, -0.367864693, 0.319238901, -0.179704017, 0.012684989]}, dtype=object),
                                                                                                                    'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_Y_001.xlsx' for n in range(2)], 'Analyte': ['Analyte 3' for n in range(2)], 'Split 2_Peptide 2': [0.117336152, -0.036997886]}, dtype=object)})
                    })}),
                    '_002': OrderedDict({'Split 1': OrderedDict({
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_002.xlsx': OrderedDict({'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_002.xlsx' for n in range(3)], 'Analyte': ['Analyte 3' for n in range(3)], 'Split 1_Peptide 2': [-14.40903226, 7.774193548, 15]}, dtype=object),
                                                                                                                    'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_002.xlsx' for n in range(3)], 'Analyte': ['Analyte 1' for n in range(3)], 'Split 1_Peptide 2': [-14.83870968, -16.88387097, 11.12903226]}, dtype=object)})
                    }),
                                         'Split 2': OrderedDict({
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_K_002.xlsx': OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_K_002.xlsx' for n in range(4)], 'Analyte': ['Analyte 1' for n in range(4)], 'Split 2_Peptide 2': [0.055445545, -0.356435644, -1.401188119, -1.328712871]}, dtype=object),
                                                                                                                    'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_K_002.xlsx' for n in range(2)], 'Analyte': ['Analyte 3' for n in range(2)], 'Split 2_Peptide 2': [-0.445544554, -1.916435644]}, dtype=object)}),
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_L_002.xlsx': OrderedDict({'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_L_002.xlsx' for n in range(3)], 'Analyte': ['Analyte 3' for n in range(3)], 'Split 2_Peptide 2': [-1.527704485, -0.490765172, -1.712401055]}, dtype=object),
                                                                                                                    'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_2/Test_plate_L_002.xlsx'], 'Analyte': ['Analyte 1'], 'Split 2_Peptide 2': [-2.100263852]}, dtype=object)})
                    })})
                })
                exp_all_features = ['Split 1_Peptide 2', 'Split 2_Peptide 2']
                exp_split_features = OrderedDict({'Split 1': ['Split 1_Peptide 2'],
                                                  'Split 2': ['Split 2_Peptide 2']})
                exp_analytes = ['Analyte 1', 'Analyte 3']
                exp_orig_fluor_data = pd.DataFrame({
                    'Split 1_Peptide 2': [-0.288463748, 0.128271115, -14.83870968, 7.774193548],
                    'Split 2_Peptide 2': [-0.493657505, -0.036997886, -1.328712871, -1.527704485],
                    'Analyte': ['Analyte 1', 'Analyte 3', 'Analyte 1', 'Analyte 3']
                })
                exp_ml_fluor_data = pd.DataFrame({
                    'Split 1_Peptide 2': [-0.288463748, -14.83870968, 0.128271115, 7.774193548],
                    'Split 2_Peptide 2': [-0.493657505, -1.328712871, -0.036997886, -1.527704485],
                    'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 3', 'Analyte 3']
                })
                exp_stats_dict = OrderedDict({'Split 1_Peptide 2': OrderedDict({'F statistic': 1.9630775054759544,
                                                                                'p value': 0.29619485006889945}),
                                              'Split 2_Peptide 2': OrderedDict({'F statistic': 0.022740970530456296,
                                                                                'p value': 0.8939686326166734})})

                # Input values
                data_dirs_dict = {'Split 1': 'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/',
                                  'Split 2': 'tests/Test_plates/Test_plate_parsing_pipeline/Split_2/'}
                results_dir = 'tests/Test_plates/Test_plate_parsing_pipeline/Temp_output/'
                repeat_labels_list = ['_001', '_002']
                peptide_dict = {'Split 1': ['Peptide 2', 'Peptide X', 'Peptide 4'],
                                'Split 2': ['Peptide 1', 'Peptide 2']}
                control_peptides = {'Split 1': ['Peptide X']}
                control_analytes = ['Analyte 2']
                gain = 1
                min_fluor = 0
                max_fluor = 15

                test_parsing = ParseArrayData(
                    data_dirs_dict, results_dir, repeat_labels_list, peptide_dict,
                    control_peptides, control_analytes, gain, min_fluor, max_fluor, True
                )

            elif n == 1:
                # Expected values
                exp_peptide_list = [
                    'Split 1_Peptide 2', 'Split 1_Peptide X', 'Split 1_Peptide 4'
                ]
                exp_repeat_dict = OrderedDict({
                    '_001': OrderedDict({'Split 1': ['Test_plate_A_001.xlsx',
                                                     'Test_plate_B_001.xlsx']}),
                    '_002': OrderedDict({'Split 1': ['Test_plate_B_002.xlsx']})
                })
                exp_plates = OrderedDict({
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_A_001.xlsx': pd.DataFrame({
                        'A': [0.469, 9.85, 1.12, 8.21, 1, 0.379],
                        'B': [13.9, 7.66, 12.2, 9.81, 9.49, 4.5],
                        'C': [9.81, 9.3, 3.42, 1.04, 7.46, 3.6],
                        'D': [9.42, 3.89, 7.45, 0.448, 14, 6.94],
                        'E': [8.1, 8.28, 4.37, 5.03, 3.4, 9.22],
                        'F': [10.6, 4.32, 14.3, 2.44, 8.17, 0.517]
                    }),
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_001.xlsx': pd.DataFrame({
                        'A': [9.63, 2.23, 3.46, 6.76, 8.15, 14.91],
                        'B': [9.63, 3.71, 8.56, 7.06, 7.9, 8.36],
                        'C': [np.nan, 4.44, 6.31, 6.31, 2.35, 1.15],
                        'D': [12.2, 8.86, 6.84, 6.57, 13.8, 7.81],
                        'E': [8.49, 6.99, 9.5, 1.14, 0.304, 8.02],
                        'F': [8.19, 8.29, 8.26, 6.84, 9.32, 9.77]
                    }),
                    'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_002.xlsx': pd.DataFrame({
                        'A': [3.07, 0.0732, 5.8, 5.58, 4.54, 9.19],
                        'B': [7.07, 7.93, 13.2, 0.568, 6.48, 7.46],
                        'C': [6.45, 1.42, 6.24, 3.26, 6.02, 0.786],
                        'D': [14.4, 0.465, 6.1, 5.09, 9, 2.19],
                        'E': [5.79, 6.95, 6.65, 6.96, 2.64, 9.47],
                        'F': [5.3, 3.29, 10.5, 3.03, 13.8, 7.01]
                    })
                })
                exp_scaled_data = OrderedDict({
                    '_001': OrderedDict({'Split 1': OrderedDict({
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_A_001.xlsx': OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_A_001.xlsx' for n in range(5)], 'Analyte': ['Analyte 1' for n in range(5)], 'Split 1_Peptide 2': [0.614730553, 0.551533954, 0.060898541, -0.47351488, -0.103412616]}, dtype=object),
                                                                                                                    'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_A_001.xlsx'], 'Analyte': ['Analyte 3'], 'Split 1_Peptide 2': [0.455015512]}, dtype=object)}),
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_001.xlsx': OrderedDict({'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_001.xlsx' for n in range(5)], 'Analyte': ['Analyte 1' for n in range(5)], 'Split 1_Peptide 2': [-0.851908397, -0.514503817, -1.018320611, -1.016793893, 0.032061069]}, dtype=object),
                                                                                                                    'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_001.xlsx'], 'Analyte': ['Analyte 3'], 'Split 1_Peptide 2': [-0.198473282]}, dtype=object)})
                    })}),
                    '_002': OrderedDict({'Split 1': OrderedDict({
                        'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_002.xlsx': OrderedDict({'Analyte 3': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_002.xlsx' for n in range(3)], 'Analyte': ['Analyte 3' for n in range(3)], 'Split 1_Peptide 2': [-14.40903226, 7.774193548, 15]}, dtype=object),
                                                                                                                    'Analyte 1': pd.DataFrame({'Plate': ['tests/Test_plates/Test_plate_parsing_pipeline/Split_1/Test_plate_B_002.xlsx' for n in range(3)], 'Analyte': ['Analyte 1' for n in range(3)], 'Split 1_Peptide 2': [-14.83870968, -16.88387097, 11.12903226]}, dtype=object)})
                    })})
                })
                exp_all_features = ['Split 1_Peptide 2']
                exp_split_features = OrderedDict({'Split 1': ['Split 1_Peptide 2']})
                exp_analytes = ['Analyte 1', 'Analyte 3']
                exp_orig_fluor_data = pd.DataFrame({
                    'Split 1_Peptide 2': [-0.288463748, 0.128271115, -14.83870968, 7.774193548],
                    'Analyte': ['Analyte 1', 'Analyte 3', 'Analyte 1', 'Analyte 3']
                })
                exp_ml_fluor_data = pd.DataFrame({
                    'Split 1_Peptide 2': [-0.288463748, -14.83870968, 0.128271115, 7.774193548],
                    'Analyte': ['Analyte 1', 'Analyte 1', 'Analyte 3', 'Analyte 3']
                })
                exp_stats_dict = OrderedDict({'Split 1_Peptide 2': OrderedDict({'F statistic': 1.9630775054759544,
                                                                                'p value': 0.29619485006889945})})

                # Input values
                data_dirs_dict = {'Split 1': 'tests/Test_plates/Test_plate_parsing_pipeline/Split_1/'}
                results_dir = 'tests/Test_plates/Test_plate_parsing_pipeline/Temp_output/'
                repeat_labels_list = ['_001', '_002']
                peptide_dict = {'Split 1': ['Peptide 2', 'Peptide X', 'Peptide 4']}
                control_peptides = {'Split 1': ['Peptide X']}
                control_analytes = ['Analyte 2']
                gain = 1
                min_fluor = 0
                max_fluor = 15

                test_parsing = ParseArrayData(
                    data_dirs_dict, results_dir, repeat_labels_list, peptide_dict,
                    control_peptides, control_analytes, gain, min_fluor, max_fluor, True
                )

            # Run the pipeline
            test_parsing.group_xlsx_repeats(ignore_files={'Split 1': ['Test_plate_A_002.xlsx']})
            test_parsing.xlsx_to_scaled_df(
                no_pep={'Split 1': 'Peptide 4', 'Split 2': 'Peptide 1'},
                scale_method='analyte_fluorophore', draw_plot=True, plot_dir_name=''
            )
            test_parsing.combine_plate_readings(same_num_repeats=False)
            test_parsing.display_data_distribution()
            act_stats_dict = test_parsing.run_anova()

            # Run at the end to check that values aren't updated during the run
            self.assertEqual(data_dirs_dict, test_parsing.data_dirs_dict)
            self.assertEqual(results_dir.rstrip('/'), test_parsing.results_dir)
            self.assertEqual(repeat_labels_list, test_parsing.repeat_list)
            self.assertEqual(peptide_dict, test_parsing.peptide_dict)
            self.assertEqual(exp_peptide_list, test_parsing.peptide_list)
            self.assertEqual(control_peptides, test_parsing.control_peptides)
            self.assertEqual(control_analytes, test_parsing.control_analytes)
            self.assertEqual(gain, test_parsing.gain)
            self.assertEqual(min_fluor, test_parsing.min_fluor)
            self.assertEqual(max_fluor, test_parsing.max_fluor)
            self.assertEqual(exp_repeat_dict, test_parsing.repeat_dict)
            self.assertEqual(list(exp_plates.keys()), list(test_parsing.plates.keys()))
            for plate_path in test_parsing.plates.keys():
                pd.testing.assert_frame_equal(
                    exp_plates[plate_path], test_parsing.plates[plate_path]
                )
            self.assertEqual(list(exp_scaled_data.keys()), list(test_parsing.scaled_data.keys()))
            for repeat in test_parsing.scaled_data.keys():
                self.assertEqual(
                    list(exp_scaled_data[repeat].keys()), list(test_parsing.scaled_data[repeat].keys())
                )
                for split in test_parsing.scaled_data[repeat].keys():
                    self.assertEqual(
                        list(exp_scaled_data[repeat][split].keys()),
                        list(test_parsing.scaled_data[repeat][split].keys())
                    )
                    for plate_path in test_parsing.scaled_data[repeat][split].keys():
                        self.assertEqual(
                            list(exp_scaled_data[repeat][split][plate_path].keys()),
                            list(test_parsing.scaled_data[repeat][split][plate_path].keys())
                        )
                        for analyte in test_parsing.scaled_data[repeat][split][plate_path]:
                            act_df = test_parsing.scaled_data[repeat][split][plate_path][analyte]
                            exp_df = exp_scaled_data[repeat][split][plate_path][analyte]
                            pd.testing.assert_frame_equal(exp_df, act_df)
            self.assertEqual(exp_all_features, test_parsing.all_features)
            self.assertEqual(exp_split_features, test_parsing.split_features)
            self.assertEqual(exp_analytes, test_parsing.analytes)
            pd.testing.assert_frame_equal(exp_orig_fluor_data, test_parsing.orig_fluor_data)
            pd.testing.assert_frame_equal(exp_ml_fluor_data, test_parsing.ml_fluor_data)
            self.assertEqual(list(exp_stats_dict.keys()), list(act_stats_dict.keys()))
            for peptide in exp_stats_dict.keys():
                self.assertEqual(['F statistic', 'p value'], list(exp_stats_dict[peptide].keys()))
                self.assertEqual(['F statistic', 'p value'], list(act_stats_dict[peptide].keys()))
                exp_F = exp_stats_dict[peptide]['F statistic']
                exp_p = exp_stats_dict[peptide]['p value']
                act_F = act_stats_dict[peptide]['F statistic']
                act_p = act_stats_dict[peptide]['p value']
                np.testing.assert_almost_equal(exp_F, act_F, 7)
                np.testing.assert_almost_equal(exp_p, act_p, 7)

            shutil.rmtree('tests/Test_plates/Test_plate_parsing_pipeline/Temp_output/')
