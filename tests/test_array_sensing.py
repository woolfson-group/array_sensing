
import unittest


class TestClass(unittest.TestCase):

    def plate_parsing(self, test, dir_path, rows_skip):
        """
        Checks that test plates are parsed as expected

        Input
        --------
        - test: The id of the test being run
        - dir_path: Path (either absolute or relative) to directory containing
        test plate data
        - rows_skip: Dictionary listing for each test the analytes on the plate and
        the location of the manually prepared data (for comparison with the results
        of running the code in parse_array_data)
        """

        import random
        import shutil
        import string
        import numpy as np
        import pandas as pd
        from subroutines.parse_array_data import ParseArrayData
        from subroutines.exceptions import NaNFluorescenceError

        repeats = ['repeat_1']
        peptides = ['No Pep', 'GRP22', 'GRP35', 'GRP46', 'GRP51', 'GRP52',
                    'GRP63', 'GRP80', 'Pent', 'Hex', 'Hex2', 'Hept', '24D',
                    '24E', '24K', '17K']
        results_dir = ''.join(random.choice(
            string.ascii_uppercase + string.ascii_lowercase + string.digits
        ) for n in range(6))
        control_peptides = ['GRP35']
        control_analytes = []
        test_peptides = [peptide for peptide in peptides
                         if not peptide in ['No Pep', 'GRP35']]

        # Parsed data
        fluor_data = ParseArrayData(
            dir_path, repeats, peptides, results_dir, control_peptides, control_analytes
        )
        fluor_data.group_xlsx_repeats()
        if test == 'Test_4':
            with self.assertRaises(NaNFluorescenceError): fluor_data.xlsx_to_scaled_df('No Pep')
            return
        else:
            fluor_data.xlsx_to_scaled_df('No Pep')
        if test == 'Test_3':
            with self.assertRaises(NaNFluorescenceError): fluor_data.combine_plate_readings()
            return
        else:
            fluor_data.combine_plate_readings()
        parsed_data = fluor_data.ml_fluor_data.set_index('Analyte')

        analytes = sorted([
            tea for tea in ['Black Clipper', 'Black Diplomat', 'Green DoubleDragon',
            'Grey Asda'] if tea in parsed_data.index.tolist()
        ])
        summary_parsed = parsed_data.loc[analytes]
        shutil.rmtree(results_dir)

        # Manually processed data
        bc_manual = pd.read_excel(
            '{}Test_plate_amalgamation.xlsx'.format(dir_path),
            skiprows=rows_skip['bc_manual']['skiprows'],
            nrows=rows_skip['bc_manual']['nrows'], index_col=0
        )
        bd_manual = pd.read_excel(
            '{}Test_plate_amalgamation.xlsx'.format(dir_path),
            skiprows=rows_skip['bd_manual']['skiprows'],
            nrows=rows_skip['bd_manual']['nrows'], index_col=0
        )
        ga_manual = pd.read_excel(
            '{}Test_plate_amalgamation.xlsx'.format(dir_path),
            skiprows=rows_skip['ga_manual']['skiprows'],
            nrows=rows_skip['ga_manual']['nrows'], index_col=0
        )
        gd_manual = pd.read_excel(
            '{}Test_plate_amalgamation.xlsx'.format(dir_path),
            skiprows=rows_skip['gd_manual']['skiprows'],
            nrows=rows_skip['gd_manual']['nrows'], index_col=0
        )
        summary_manual = pd.read_excel(
            '{}Test_plate_amalgamation.xlsx'.format(dir_path),
            skiprows=rows_skip['summary_manual']['skiprows'],
            nrows=rows_skip['summary_manual']['nrows'], index_col=0
        )

        paired_teas = {'Black Clipper': [bc_manual],
                       'Black Diplomat': [bd_manual],
                       'Grey Asda': [ga_manual],
                       'Green DoubleDragon': [gd_manual]}
        for tea in paired_teas.keys():
            dfs = []
            for df_list in fluor_data.scaled_data['repeat_1']:
                dfs.append(df_list[tea])
            df = pd.concat(dfs, axis=0).reset_index(drop=True)
            df = df[test_peptides]
            paired_teas[tea].append(df)

        for col in test_peptides:
            for df_list in paired_teas.values():
                for df in df_list:
                    if col in df.columns:
                        non_nan_vals = []
                        nan_vals = []
                        for val in df[col].tolist():
                            if np.isnan(val):
                                nan_vals.append(val)
                            else:
                                non_nan_vals.append(val)
                        rows = nan_vals + sorted(non_nan_vals)
                        df[col] = rows

        for tea in paired_teas.keys():
            np.testing.assert_array_almost_equal(
                paired_teas[tea][0].values, paired_teas[tea][1].values, decimal=6
            )
        np.testing.assert_array_almost_equal(
            summary_manual.values, summary_parsed.values, decimal=6
        )

    def test_plate_parsing(self):
        """
        """

        import os

        tests = ['Test_1', 'Test_2', 'Test_3', 'Test_4']
        dir_paths = {test: 'tests/Test_plates/{}/'.format(test) for test in tests}
        rows_skip = {'Test_1': {'bc_manual': {'nrows': 20, 'skiprows': 2},
                                'bd_manual': {'nrows': 20, 'skiprows': 26},
                                'gd_manual': {'nrows': 20, 'skiprows': 50},
                                'ga_manual': {'nrows': 20, 'skiprows': 74},
                                'summary_manual': {'nrows': 4, 'skiprows': 98}},
                     'Test_2': {'bc_manual': {'nrows': 10, 'skiprows': 2},
                                'bd_manual': {'nrows': 10, 'skiprows': 16},
                                'gd_manual': {'nrows': 10, 'skiprows': 30},
                                'ga_manual': {'nrows': 10, 'skiprows': 44},
                                'summary_manual': {'nrows': 4, 'skiprows': 58}},
                     'Test_3': {'bc_manual': {'nrows': 5, 'skiprows': 2},
                                'bd_manual': {'nrows': 5, 'skiprows': 11},
                                'gd_manual': {'nrows': 5, 'skiprows': 20},
                                'ga_manual': {'nrows': 5, 'skiprows': 29},
                                'summary_manual': {'nrows': 4, 'skiprows': 38}},
                     'Test_4': {'bc_manual': {'nrows': 5, 'skiprows': 2},
                                'bd_manual': {'nrows': 5, 'skiprows': 11},
                                'gd_manual': {'nrows': 5, 'skiprows': 20},
                                'ga_manual': {'nrows': 5, 'skiprows': 29},
                                'summary_manual': {'nrows': 4, 'skiprows': 38}}}
        for test in tests:
            self.plate_parsing(test, dir_paths[test], rows_skip[test])

    def test_ml(self):
        """
        """

        import sklearn
        from subroutines.train import RunML
