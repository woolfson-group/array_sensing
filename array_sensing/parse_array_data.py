

# Functions to parse data collected from plate reader and manipulate it into a
# suitable format to be fed into sklearn ML algorithms.

import copy
import os
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing

if __name__ == 'array_sensing.parse_array_data':
    from array_sensing.exceptions import (
        PlateLayoutError, FluorescenceSaturationError
    )
else:
    from sensing_array_paper.array_sensing.exceptions import (
        PlateLayoutError, FluorescenceSaturationError
    )


def trim_dataframe(label_df):
    """
    """
    row_index = ''
    for index in range(label_df.shape[0]):
        row_set = set(label_df.iloc[index])
        if all(isinstance(x, (int, float)) for x in row_set):
            if all(np.isnan(list(row_set)[int(x)]) for x in range(len(row_set))):
                row_index = index
                break
    if row_index == '':
        row_index = label_df.shape[0]

    col_index = ''
    for index, col in enumerate(label_df.columns):
        col_set = set(label_df[col].tolist()[0:row_index])
        if all(isinstance(x, (int, float)) for x in col_set):
            if all(np.isnan(list(col_set)[int(x)]) for x in range(len(col_set))):
                col_index = index
                break
    if col_index == '':
        col_index = label_df.shape[1]

    label_df = label_df.iloc[:row_index, :col_index].reset_index(drop=True)

    return label_df


def parse_xlsx_to_dataframe(plate_path, peptide_list, gain):
    """
    Converts an excel xlsx file output from the plate reader into a pandas
    dataframe.

    The layout of analytes and peptides on the plate is read from the "Plate
    Information" sheet in the xlsx file, and used to group r x c blocks of
    fluorescence readings collected for the same analyte in the "End Point"
    sheet.

    Input
    ----------
    - plate_path: File path to input xlsx file
    - peptide_list: List of barrel names
    - gain: Which dataframe of fluorescence data (collected at different gains)
    to use

    Output
    ----------
    - grouped_fluor_data: Dictionary of dataframes of fluorescence readings for
    each analyte
    """

    # Determines which table of fluorescence data (collected at different
    # gains with the fluorimeter) to use. By default uses the first listed.
    if (
           (type(gain) != int)
        or (type(gain) == int and gain <= 0)
    ):
        raise TypeError('Gain value not recognised. Please specify a positive'
                        ' integer')
    skiprows = (gain*19)-5
    print('Parsing plate {}'.format(plate_path))

    # Reads fluorescence data from "End point" sheet
    fluor_df = pd.read_excel(
        plate_path, sheet_name='End point', skiprows=skiprows, nrows=16,
        index_col=0, usecols=range(25)
    ).reset_index(drop=True)

    # Reads class labels from metadata in "Protocol Information" sheet
    protocol_df = pd.read_excel(
        plate_path, sheet_name='Protocol Information', skiprows=0, index_col=0
    )
    protocol_df_index = [str(x).lower() for x in protocol_df.index.tolist()]
    start_row = protocol_df_index.index('plate layout')
    label_df = pd.read_excel(
        plate_path, sheet_name='Protocol Information', skiprows=start_row+2,
        index_col=0
    )

    label_df = trim_dataframe(label_df)
    label_df = label_df.replace(
        to_replace='[bB][lL][aA][nN][kK]', value='blank', regex=True
    )

    label_array = copy.deepcopy(label_df).to_numpy()
    label_set = list(set(label_array.flatten('C')))

    # Reads peptide layout and names from metadata in "Protocol Information"
    # sheet
    start_row = protocol_df_index.index('peptide layout')
    peptide_arrang = pd.read_excel(
        plate_path, sheet_name='Protocol Information', skiprows=start_row+1,
        index_col=None
    )
    peptide_arrang = trim_dataframe(peptide_arrang)

    r_dim = peptide_arrang.shape[0]
    c_dim = peptide_arrang.shape[1]
    plate_peptides = copy.deepcopy(peptide_arrang).to_numpy().flatten('F').tolist()

    # Checks that peptides listed in the xlsx file are the same as those input
    # by the user
    if sorted(peptide_list) != sorted(plate_peptides):
        raise PlateLayoutError('Peptides tested in {} don\'t match peptides '
                               'specified by user'.format(plate_path))
    if len(set(plate_peptides)) != (r_dim * c_dim):
        multi_peptides = []
        for peptide in plate_peptides:
            if plate_peptides.count(peptide) > 1 and not peptide in multi_peptides:
                multi_peptides.append(peptide)
        raise PlateLayoutError(
            'Barrel(s) {} listed more than once in plate layout listed in {}'.format(
                multi_peptides, plate_path
            )
        )

    # Combines fluorescence data collected for the same analyte into a
    # dataframe
    grouped_fluor_data = {}

    for analyte in label_set:
        grouped_fluor_data[analyte] = []

    for index, analyte in np.ndenumerate(label_array):
        index_r = index[0]
        index_c = index[1]

        rows = np.arange((r_dim*index_r), ((r_dim*index_r)+(r_dim+1)))
        # + (r+1) so that row range covers r rows
        columns = np.arange(c_dim*(index_c), ((c_dim*index_c)+(c_dim+1)))
        # + (c+1) so that column range covers 2 columns

        sub_fluor_df = fluor_df.iloc[rows[0]:rows[-1], columns[0]:columns[-1]]
        sub_fluor_vals = copy.deepcopy(sub_fluor_df).to_numpy().flatten('F')

        ordered_fluor_vals = [plate_path, analyte]
        count = len(grouped_fluor_data[analyte])+1
        for peptide in peptide_list:
            r_loc, c_loc = (copy.deepcopy(peptide_arrang).to_numpy() == peptide).nonzero()
            r_loc = r_loc[0]  # Already checked that peptide is present once only above
            c_loc = c_loc[0]
            fluor_reading = sub_fluor_df.iloc[r_loc, c_loc]
            orig_posn = [(r_dim*index_r)+r_loc, (c_dim*index_c)+c_loc]
            ordered_fluor_vals += [fluor_reading, orig_posn]
        grouped_fluor_data[analyte].append(pd.DataFrame({count: ordered_fluor_vals}))

    for analyte in label_set:
        analyte_df = pd.concat(
            grouped_fluor_data[analyte], axis=1, ignore_index=True
        ).transpose().reset_index(drop=True)
        analyte_columns = ['Plate', 'Analyte']
        for peptide in peptide_list:
            analyte_columns += [peptide, '{}_loc'.format(peptide)]
        analyte_df.columns = analyte_columns
        grouped_fluor_data[analyte] = analyte_df

    return fluor_df, grouped_fluor_data


def check_for_saturation(
    saturated_readings, plate, plate_path, peptide_list, min_fluor, max_fluor
):
    """
    Checks that no fluorescence readings are outside of the fluorescence range
    that can be measured by the plate reader.

    Input
    ----------
    - saturated_readings: List of saturated measurements, identified by plate
    file path and barrel name
    - plate: Dictionary of dataframes of fluorescence readings from a plate
    - plate_path: File path to the xlsx file from which the plate dictionary
    was parsed
    - peptide_list: List of barrel names
    - min_fluor: Minimum fluoresence reading the plate can measure
    - max_fluor: Maximum fluorescence reading the plate can measure

    Output
    ----------
    - saturated_readings: Updated list of saturated measurements, identified by
    plate file path and barrel name
    """

    for analyte, fluor_df in plate.items():
        for index, val in np.ndenumerate(copy.deepcopy(fluor_df).to_numpy()):
            r_loc = index[0]
            c_loc = index[1]
            column = fluor_df.columns[c_loc]
            if (
                    column in peptide_list
                and (val <= min_fluor or val >= max_fluor)
            ):
                loc_column = fluor_df.columns.tolist().index('{}_loc'.format(column))
                raw_data_loc = fluor_df.iloc[r_loc, loc_column]
                data_point = [plate_path, analyte, column, val, raw_data_loc[0], raw_data_loc[1]]
                saturated_readings.append(data_point)

    return saturated_readings


def color_val_red(input_df):
    """
    Colours text of selected cells red in dataframe when displayed in jupyter
    notebook
    """
    return 'color: red'

def highlight_yellow(input_df):
    """
    Highlights background of selected cells yellow in dataframe when displayed
    in jupyter notebook
    """
    return 'background-color: yellow'


def calc_median_and_remove_outliers(
    plate_df, raw_plate_data, plate_outliers, stage, alpha=0.05
):
    """
    Calculates the median of each column in a dataframe (ignoring data points
    identified as outliers by a generalised ESD test)
    http://finzi.psych.upenn.edu/R/library/EnvStats/html/rosnerTest.html

    Input
    ----------
    - input_df: DataFrame, dimensions = n_samples x m_features

    Output
    ----------
    - output_ser: Series, dimensions = n_features
    """

    import math
    from scipy import stats

    features = [feature for feature in plate_df.columns if feature not in
                ['Plate', 'Analyte'] and not feature.endswith('_loc')]
    drop_columns = [feature for feature in plate_df.columns
                    if feature not in features]
    for feature in features:
        if set(pd.isnull(plate_df[feature])) == {True}:
            continue

        feature_array = copy.deepcopy(plate_df[feature]).to_numpy()
        n = feature_array.shape[0]
        if n < 15:
            k_max = 1
        elif 15 <= n < 25:
            k_max = 2
        else:
            k_max = math.floor(n / 2)

        k = 0
        while k < k_max:
            # Calculation of Grubbs test threshold value
            t = stats.t.ppf((1 - (alpha / (2*(n-k)))), (n-k-2))
            numerator = (n-k-1) * t
            denominator = np.sqrt(n-k) * np.sqrt((n-k-2)+np.square(t))
            g_critical = numerator / denominator

            # Calculation of Grubbs test statistic
            abs_diff = np.abs(feature_array - np.nanmean(feature_array))
            max_val = np.nanmax(abs_diff, axis=0)
            max_index = np.nanargmax(abs_diff, axis=0)
            g_calculated = max_val / np.nanstd(feature_array, ddof=1)  # Uses sample standard deviation as required by test

            if g_calculated > g_critical:
                feature_array[max_index] = np.nan
                k += 1

                plate_name = plate_df['Plate'][max_index]
                analyte = plate_df['Analyte'][max_index]
                raw_data_loc = plate_df['{}_loc'.format(feature)][max_index]
                r = raw_data_loc[0]
                c = raw_plate_data[plate_name].columns.tolist()[raw_data_loc[1]]

                if not plate_name in list(plate_outliers.keys()):
                    plate_outliers[plate_name] = {}
                plate_outliers[plate_name]['{}_{}_{}_{}'.format(
                    analyte, feature, raw_data_loc[0], raw_data_loc[1]
                )] = max_val

                if stage == 1:
                    raw_plate_data[plate_name].applymap(
                        color_val_red, subset=pd.IndexSlice[r, [c]]
                    )
                elif stage == 2:
                    raw_plate_data[plate_name].applymap(
                        highlight_yellow, subset=pd.IndexSlice[r, [c]]
                    )
                else:
                    raise ValueError('Inappropriate value provided for '
                                     'stage variable - should be set equal '
                                     'to 1 or 2')
            else:
                break

    median_df = plate_df.drop(drop_columns, axis=1)
    median_df = median_df.median().to_frame().transpose()

    return median_df, raw_plate_data, plate_outliers


def scale_min_max(
    plate, no_pep, plate_name, cols_ignore, raw_plate_data, plate_outliers
):
    """
    Scales data on plate between min fluorescence reading (analyte + no peptide
    + DPH) and max fluorescence reading (= blank readings, i.e. no analyte +
    peptide + DPH)

    Input
    ----------
    - plate: Dictionary of dataframes of plate fluorescence data
    - no_pep: Name of sample without peptide in xlsx files of fluorescence data

    Output
    ----------
    - scaled_plate: Dictionary of dataframes of min max scaled fluorescence
    readings for each analyte
    """

    try:
        blank_data = copy.deepcopy(plate['blank'])
    except KeyError:
        raise PlateLayoutError('No blank readings (= no analyte + peptide + '
                               'fluorophore) included on plate')
    blank_data, plate_orig_data, plate_outliers = calc_median_and_remove_outliers(
        blank_data, raw_plate_data, plate_outliers, stage=1
    )
    if any(np.isnan(x) for x in blank_data.values.flatten()):
        nan_peptides = [col for col in blank_data.columns if np.isnan(blank_data[col][0])]
        sys.exit('ERROR: Median flourescence reading in the absence of analyte on '
                 'plate {} is NaN for barrels {}'.format(plate_name, nan_peptides))

    scaled_plate = {}
    analytes = [analyte for analyte in list(plate.keys()) if analyte != 'blank']
    for analyte in analytes:
        fluor_data = copy.deepcopy(plate[analyte])

        # Calculates minimum fluorescence reading for the analyte
        min_fluor, raw_plate_data, plate_outliers = calc_median_and_remove_outliers(
            fluor_data[['Plate', 'Analyte', no_pep, '{}_loc'.format(no_pep)]],
            raw_plate_data, plate_outliers, stage=1
        )
        min_fluor = min_fluor[no_pep][0]
        if np.isnan(min_fluor):
            sys.exit('ERROR: Median flourescence reading for no peptide on plate '
                     '{} is NaN for analyte {}'.format(plate_name, analyte))

        fluor_data = fluor_data.drop([no_pep, '{}_loc'.format(no_pep)], axis=1)

        for index_c, column in enumerate(list(fluor_data.columns)):
            if (
                    not column.endswith('_loc')
                and not column in ['Plate', 'Analyte', no_pep]
                and not column in cols_ignore
            ):
                # Applies min max scaling to each reading
                max_fluor = blank_data[column][0]

                for index_r, val in enumerate(fluor_data[column].tolist()):
                    if (max_fluor - min_fluor) > 0:
                        scaled_val = (val-min_fluor) / (max_fluor-min_fluor)
                    else:
                        scaled_val = np.nan
                        if max_fluor == min_fluor:
                            print('\x1b[31m WARNING - min and max fluorescence '
                                  'readings for analyte {} peptide {} on plate '
                                  '{} are the same \033[0m'.format(analyte,
                                  column, fluor_data['Plate'][index_r]))
                        elif max_fluor < min_fluor:
                            if max_fluor == min_fluor:
                                print('\x1b[31m WARNING - median max. fluorescence'
                                      ' reading for analyte {} peptide {} on plate '
                                      '{} is larger than the corresponding median '
                                      'min. fluorescene reading \033[0m'.format(
                                      analyte, column, fluor_data['Plate'][index_r]))
                    fluor_data.iloc[index_r, index_c] = scaled_val

        drop_columns = []
        for column in cols_ignore:
            drop_columns += [column, '{}_loc'.format(column)]
        fluor_data = fluor_data.drop(drop_columns, axis=1)

        scaled_plate[analyte] = fluor_data

    return scaled_plate, raw_plate_data, plate_outliers


class DefData():

    def __init__(self, dir_path, repeat_names, peptide_list, results_dir):
        """
        - dir_path: Path (either absolute or relative) to directory containing
        xlsx files of fluorescence readings
        - repeat_names: Names used to label different repeat readings of the
        same analytes in the xlsx file names. **NOTE: the same repeat name
        should be used for all analytes in the same repeat (hence date is
        probably not the best label unless all analytes in the repeat run were
        measured on the same day)**.
        - peptide_list: List of barrel names
        """

        self.dir_path = dir_path
        self.repeats = repeat_names
        self.peptides = peptide_list

        if not os.path.isdir(self.dir_path):
            raise FileNotFoundError('Path to working directory not recognised')
        if len(self.peptides) != len(set(self.peptides)):
            multi_peptides = []
            for peptide in self.peptides:
                if self.peptides.count(peptide) > 1 and not peptide in multi_peptides:
                    multi_peptides.append(peptide)
            raise PlateLayoutError(
                'Barrel(s) listed more than once: {}'.format(multi_peptides)
            )

        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        else:
            print('Directory {} already found in {}'.format(results_dir, os.getcwd()))
            remove = ''
            while not isinstance(remove, bool):
                remove = input('Overwrite {}?'.format(results_dir))
                if remove.strip().lower() in ['true', 'yes', 'y']:
                    remove = True
                elif remove.strip().lower() in ['false', 'no', 'n']:
                    remove = False
                else:
                    print('Input not recognised - please specify "yes" or "no"')
                    remove = ''
            if remove is True:
                shutil.rmtree(results_dir)
                os.mkdir(results_dir)
        self.results_dir = results_dir.rstrip('/')


class ParseArrayData(DefData):


    def __init__(
        self, dir_path, repeat_names, peptide_list, results_dir,
        control_peptides, control_analytes, gain=1, min_fluor=0,
        max_fluor=260000
    ):
        """
        - dir_path: Path (either absolute or relative) to directory containing
        xlsx files of fluorescence readings
        - repeat_names: Names used to label different repeat readings of the
        same analytes in the xlsx file names. **NOTE: the same repeat name
        should be used for all analytes in the same repeat (hence date is
        probably not the best label unless all analytes in the repeat run were
        measured on the same day)**.
        - peptide_list: List of barrel names
        - gain: Which dataframe of fluorescence data (collected at different
        gains) to use, default is first listed in xlsx file
        - min_fluor: Minimum fluorescence reading that can be measured by the
        plate reader, default is 0
        - max_fluor: Maximum fluorescence reading that can be measured by the
        plate reader, default is 260,000
        """
        DefData.__init__(self, dir_path, repeat_names, peptide_list, results_dir)

        self.gain = gain
        self.min_fluor = min_fluor
        self.max_fluor = max_fluor

        self.control_peptides = control_peptides
        missing_peptides = [peptide for peptide in self.control_peptides
                            if not peptide in self.peptides]
        if len(missing_peptides) >= 1:
            raise NameError(
                'Peptides {} are not included in list of all peptides'.format(
                    [peptide for peptide in missing_peptides]
                )
            )

        self.control_analytes = control_analytes

    def group_xlsx_repeats(self):
        """
        Groups xlsx files in the directory self.dir_path into independent
        measurements (via the name of the measurement included in the file name)
        """

        ind_meas_xlsx = {}

        for repeat in self.repeats:
            ind_meas_xlsx[repeat] = []

            for file_name in os.listdir(self.dir_path):
                if (
                        not file_name.startswith('~$')
                    and file_name.endswith('{}.xlsx'.format(repeat))
                ):
                    ind_meas_xlsx[repeat].append(file_name)

        self.xlsx_grouped_by_repeat = ind_meas_xlsx

    def xlsx_to_scaled_df(self, no_pep):
        """
        Wrapper function to parse grouped xlsx files into dataframes and
        perform min max scaling.

        Input
        --------
        - no_pep: Name of sample without peptide in peptide layout
        """

        raw_plate_data = {}
        raw_plate_data_highlighted = {}
        scaled_data = {}
        outliers = {}
        saturated_readings = []

        for repeat, xlsx_list in self.xlsx_grouped_by_repeat.items():
            scaled_data[repeat] = []  # List rather than dictionary because
            # analyte name(s) is/are read from the plate not the file name

            for xlsx in xlsx_list:
                plate_path = self.dir_path + xlsx
                raw_data, plate = parse_xlsx_to_dataframe(
                    plate_path, self.peptides, self.gain
                )
                raw_plate_data[plate_path] = raw_data
                raw_plate_data_highlighted[plate_path] = copy.deepcopy(raw_data).style
                saturated_readings = check_for_saturation(
                    saturated_readings, plate, plate_path, self.peptides,
                    self.min_fluor, self.max_fluor
                )
                scaled_plate, raw_plate_data_highlighted, outliers = scale_min_max(
                    plate, no_pep, plate_path, self.control_peptides,
                    raw_plate_data_highlighted, outliers
                )
                scaled_data[repeat].append(scaled_plate)

        if saturated_readings != []:
            saturated_readings = ['{} {} {} {} (row = {}, column = {})'.format(
                    x[0], x[1], x[2], x[3], x[4], x[5]
                ) for x in saturated_readings]
            saturated_readings = '\n'.join(saturated_readings)
            raise FluorescenceSaturationError(
                'The following data points appear to have saturated the plate '
                'reader:\n{}\nTo proceed with analysis you need to either remove'
                ' the identified barrels, or replace the fluorescence\nreadings'
                ' for this barrel with readings collected at a different gain, '
                'for ALL xlsx files in the dataset.'.format(saturated_readings))

        # Checks that there is data for the same set of analytes in each repeat
        count = 0
        analytes = []
        for repeat in scaled_data.keys():
            count += 1
            repeat_analytes = [analyte for plate in scaled_data[repeat]
                               for analyte in plate.keys()]
            if count == 1:
                repeat_1 = repeat
                analytes = [analyte for analyte in repeat_analytes]
            else:
                diff_analytes = set(repeat_analytes).symmetric_difference(analytes)
                if diff_analytes != set():
                    print('\x1b[31m WARNING: The analytes tested in different '
                          'repeats are not the same. Analytes {}  are not '
                          'consistent between {} and {} \033[0m'.format(
                          diff_analytes, repeat_1, repeat))

        # Removes analytes the user has specified to ignore in the analysis
        for analyte in self.control_analytes:
            for repeat_n in scaled_data.keys():
                amalg_analytes = []
                plates = copy.deepcopy(scaled_data[repeat_n])
                for index, plate in enumerate(plates):
                    amalg_analytes += [x for x in plate.keys()
                                       if not x in amalg_analytes]
                    if analyte in plate.keys():
                        del scaled_data[repeat_n][index][analyte]
                if not analyte in amalg_analytes:
                    print('\x1b[31m WARNING: Whlist removing analytes specified'
                          ' by the user to ignore, analyte {} is not recognised'
                          ' in repeat {}'.format(analyte, repeat_n))

        self.plates = raw_plate_data
        self.same_plate_outliers = raw_plate_data_highlighted
        self.scaled_data = scaled_data
        features = [no_pep] + self.control_peptides
        self.features = [feature for feature in self.peptides if not
                         feature in features]

        with open('{}/Outliers_same_plate.txt'.format(self.results_dir), 'w') as f:
            f.write('Outliers on same plate identified by generalised ESD test:\n')
            for plate_path in list(outliers.keys()):
                for loc, outlier in outliers[plate_path].items():
                    f.write('{}: {} {}\n'.format(plate_path, loc, outlier))

    def combine_plate_readings(self):
        """
        Combines independent measurements into a single array
        """

        ml_fluor_data = []
        labels = []
        outliers = {}
        raw_plate_data = copy.deepcopy(self.same_plate_outliers)

        for repeat, plate_list in self.scaled_data.items():
            analytes = set([analyte for plate in plate_list for analyte in
                            list(plate.keys())])

            for analyte in analytes:
                # Groups dataframes measuring the same analyte together
                analyte_dfs = []

                for plate in plate_list:
                    if analyte in list(plate.keys()):
                        analyte_dfs.append(copy.deepcopy(plate[analyte]))

                analyte_df = pd.concat(analyte_dfs, axis=0).reset_index(drop=True)
                scaled_merged_data, raw_plate_data, outliers = calc_median_and_remove_outliers(
                    analyte_df, raw_plate_data, outliers, stage=2
                )

                ml_fluor_data.append(scaled_merged_data)
                labels.append(analyte)

        ml_fluor_df = pd.concat(ml_fluor_data, axis=0).reset_index(drop=True)
        if True in pd.isnull(copy.deepcopy(ml_fluor_df)).values:
            dropped_rows = [n for n in range(ml_fluor_df.shape[0])
                            if not n in ml_fluor_df.dropna(axis=0).index]
            print('\x1b[31m WARNING: readings dropped because they contain NaN '
                  'values after data processing.\nCheck plates listed in '
                  'previous warnings to make sure that all readings are '
                  'present and that min and max fluorescence readings aren\'t '
                  'identical \033[0m')
            ml_fluor_df = ml_fluor_df.dropna(axis=0).reset_index(drop=True)  # Must be after calculation of dropped_rows!
            labels = [label for index, label in enumerate(labels)
                      if not index in dropped_rows]
        labels_df = pd.DataFrame({'Analyte': labels})
        ml_fluor_df = pd.concat([ml_fluor_df, labels_df], axis=1).reset_index(drop=True)
        self.ml_fluor_data = ml_fluor_df
        self.cross_plate_outliers = raw_plate_data

        with open('{}/Outliers_across_plates.txt'.format(self.results_dir), 'w') as f:
            f.write('Outliers across merged plates identified by generalised ESD test:\n')
            for plate in list(outliers.keys()):
                for loc, outlier in outliers[plate].items():
                    f.write('{}: {} {}\n'.format(plate, loc, outlier))

    def standardise_readings(self):
        """
        Standardises fluorescence data across features (i.e. the peptides)
        """

        columns = self.ml_fluor_data.columns
        analyte_labels = self.ml_fluor_data.loc[:, 'Analyte']

        ml_fluor_data = self.ml_fluor_data.drop('Analyte', axis=1)
        standardised_ml_fluor_data = preprocessing.scale(
            ml_fluor_data.to_numpy(), axis=0, copy=True
        )  # Normalises (by subtracting the mean and dividing by the standard
        # deviation) the readings for each peptide, as required by several ML
        # algorithms
        standardised_ml_fluor_df = pd.concat(
            [pd.DataFrame(standardised_ml_fluor_data), analyte_labels], axis=1
        )
        standardised_ml_fluor_df.columns = columns

        self.standardised_ml_fluor_data = standardised_ml_fluor_df
