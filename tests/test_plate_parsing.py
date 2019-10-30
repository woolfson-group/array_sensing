
import unittest

class TestClass(unittest.TestCase):

    def test_plate_parsing(self):
        """
        """

        import random
        import shutil
        import string
        import numpy as np
        import pandas as pd
        from array_sensing.parse_array_data import ParseArrayData

        dir_path = 'tests/Test_plates/'
        repeats = ['repeat_1']
        peptides = ['No Pep', 'GRP22', 'GRP35', 'GRP46', 'GRP51', 'GRP52',
                    'GRP63', 'GRP80', 'Pent', 'Hex', 'Hex2', 'Hept', '24D',
                    '24E', '24K', '17K']
        results_dir = ''.join(random.choice(
            string.ascii_uppercase + string.ascii_lowercase + string.digits
        ) for n in range(6))
        control_peptides = ['GRP35']
        test_peptides = [peptide for peptide in peptides
                         if not peptide in ['No Pep', 'GRP35']]

        # Parsed data
        fluor_data = ParseArrayData(
            dir_path, repeats, peptides, results_dir, control_peptides
        )
        fluor_data.group_xlsx_repeats()
        fluor_data.xlsx_to_scaled_df('No Pep')
        fluor_data.combine_plate_readings()
        fluor_data.standardise_readings()
        parsed_data = fluor_data.ml_fluor_data.set_index('Analyte')
        standardised_parsed_data = fluor_data.standardised_ml_fluor_data.set_index('Analyte')
        summary_parsed = parsed_data.loc[
            ['Black Clipper', 'Black Diplomat', 'Grey Asda', 'Green DoubleDragon']
        ]
        standardised_parsed = standardised_parsed_data.loc[
            ['Black Clipper', 'Black Diplomat', 'Grey Asda', 'Green DoubleDragon']
        ]
        shutil.rmtree(results_dir)

        # Manually processed data
        bc_manual = pd.read_excel(
            'tests/Test_plates/Test_plate_amalgamation.xlsx', skiprows=2,
            nrows=20, index_col=0
        )
        bd_manual = pd.read_excel(
            'tests/Test_plates/Test_plate_amalgamation.xlsx', skiprows=26,
            nrows=20, index_col=0
        )
        ga_manual = pd.read_excel(
            'tests/Test_plates/Test_plate_amalgamation.xlsx', skiprows=50,
            nrows=20, index_col=0
        )
        gd_manual = pd.read_excel(
            'tests/Test_plates/Test_plate_amalgamation.xlsx', skiprows=74,
            nrows=20, index_col=0
        )
        summary_manual = pd.read_excel(
            'tests/Test_plates/Test_plate_amalgamation.xlsx', skiprows=98,
            nrows=4, index_col=0
        )
        standardised_manual = pd.read_excel(
            'tests/Test_plates/Test_plate_amalgamation.xlsx', skiprows=105,
            nrows=4, index_col=0
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
                    df[col] = sorted(df[col])

        for tea in paired_teas.keys():
            np.testing.assert_array_almost_equal(
                paired_teas[tea][0].values, paired_teas[tea][1].values, decimal=6
            )
        np.testing.assert_array_almost_equal(
            summary_manual.values, summary_parsed.values, decimal=6
        )
        np.testing.assert_array_almost_equal(
            standardised_manual.values, standardised_parsed.values, decimal=6
        )
