
import unittest


def plate_parsing(dir_path, rows_skip):
    """
    """

    import random
    import shutil
    import string
    import numpy as np
    import pandas as pd
    from array_sensing.parse_array_data import ParseArrayData

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

    analytes = sorted([
        tea for tea in ['Black Clipper', 'Black Diplomat', 'Green DoubleDragon',
        'Grey Asda'] if tea in parsed_data.index.tolist()
    ])
    summary_parsed = parsed_data.loc[analytes]
    standardised_parsed = standardised_parsed_data.loc[analytes]
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
    standardised_manual = pd.read_excel(
        '{}Test_plate_amalgamation.xlsx'.format(dir_path),
        skiprows=rows_skip['standardised_manual']['skiprows'],
        nrows=rows_skip['standardised_manual']['nrows'], index_col=0
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
    np.testing.assert_array_almost_equal(
        standardised_manual.values, standardised_parsed.values, decimal=6
    )


class TestClass(unittest.TestCase):

    def test_plate_parsing(self):
        """
        """

        import os

        tests = ['Test_1', 'Test_2']
        dir_paths = {test: 'tests/Test_plates/{}/'.format(test) for test in tests}
        rows_skip = {'Test_1': {'bc_manual': {'nrows': 20, 'skiprows': 2},
                                'bd_manual': {'nrows': 20, 'skiprows': 26},
                                'gd_manual': {'nrows': 20, 'skiprows': 50},
                                'ga_manual': {'nrows': 20, 'skiprows': 74},
                                'summary_manual': {'nrows': 4, 'skiprows': 98},
                                'standardised_manual': {'nrows': 4, 'skiprows': 105}},
                     'Test_2': {'bc_manual': {'nrows': 10, 'skiprows': 2},
                                'bd_manual': {'nrows': 10, 'skiprows': 16},
                                'gd_manual': {'nrows': 10, 'skiprows': 30},
                                'ga_manual': {'nrows': 10, 'skiprows': 44},
                                'summary_manual': {'nrows': 3, 'skiprows': 72},
                                'standardised_manual': {'nrows': 4, 'skiprows': 79}}}
        for test in tests:
            plate_parsing(dir_paths[test], rows_skip[test])

    def test_ml(self):
        """
        """

        import sklearn
        from array_sensing.train import RunML
