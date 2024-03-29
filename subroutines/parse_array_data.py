

# Functions to parse data collected from plate reader and manipulate it into a
# suitable format to be fed into sklearn ML algorithms.

import copy
import math
import os
import pickle
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from scipy import stats
from scipy.stats import f_oneway
from sklearn.preprocessing import RobustScaler
from xlrd import XLRDError

if __name__ == 'subroutines.parse_array_data':
    from subroutines.exceptions import (
        PlateLayoutError, FluorescenceSaturationError, NaNFluorescenceError,
        MinMaxFluorescenceError
    )
else:
    from array_sensing.subroutines.exceptions import (
        PlateLayoutError, FluorescenceSaturationError, NaNFluorescenceError,
        MinMaxFluorescenceError
    )


def trim_dataframe(label_df, plate_path):
    """
    Works out the boundaries of dataframes parsed from input xlsx files (by
    identifying rows/columns of entirely np.nan vaues), and removes excess rows
    and columns

    Input
    ----------
    - label_df: DataFrame of analyte / peptide labels parsed from an xlsx file
    - plate_path: File path to input xlsx file

    Output
    ----------
    - label_df: Input DataFrame that has been trimmed to remove excess rows
    and/or columns appended to its bottom and/or right hand side
    """

    if label_df.empty is True:
        raise PlateLayoutError(
            'Empty dataframe passed to trim_dataframe from {}'.format(plate_path)
        )

    label_df = label_df.reset_index(drop=True)

    row_index = ''
    for index in range(label_df.shape[0]):
        row_set = set(label_df.iloc[index])
        if all(isinstance(x, (int, float)) for x in row_set):
            if all(np.isnan(list(row_set)[int(x)]) for x in range(len(row_set))):
                row_index = index
                break
    if row_index == '':
        row_index = label_df.shape[0]
    if row_index == 0:
        raise PlateLayoutError('Failed to parse {} metadata'.format(plate_path))

    col_index = ''
    for index, col in enumerate(label_df.columns):
        col_set = set(label_df[col].tolist()[0:row_index])
        if all(isinstance(x, (int, float)) for x in col_set):
            if all(np.isnan(list(col_set)[int(x)]) for x in range(len(col_set))):
                col_index = index
                break
    if col_index == '':
        col_index = label_df.shape[1]
    if col_index == 0:
        raise PlateLayoutError('Failed to parse {} metadata'.format(plate_path))

    label_df = label_df.iloc[:row_index, :col_index].reset_index(drop=True)

    return label_df


def parse_xlsx_to_dataframe(plate_path, split, peptide_dict, gain=1):
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
    - split: Name of split, which has been added as a prefix to all of the
    peptide names in the plate columns
    - peptide_dict: Dictionary of lists of all peptides to be included in the
    analysis for each split
    - gain: Which dataframe of fluorescence data (collected at different gains)
    to use

    Output
    ----------
    - fluor_df: DataFrame of fluorescence readings directly parsed from the
    input xlsx file (i.e. with no further processing or reorganisation)
    - grouped_fluor_data: Dictionary of dataframes of fluorescence readings for
    each analyte
    - peptide_list: List of peptides included on the input plate
    """

    print('Parsing plate {}'.format(plate_path))

    # Determines which table of fluorescence data (collected at different
    # gains with the fluorimeter) to use. By default uses the first listed.
    if type(gain) != int:
        raise TypeError(
            'Gain value not recognised. Please specify a positive integer'
        )
    elif (type(gain) == int) and (gain <= 0):
        raise ValueError(
            'Gain value not recognised. Please specify a positive integer'
        )

    if not os.path.isfile(plate_path):
        raise FileNotFoundError('File {} doesn\'t exist'.format(plate_path))

    # Reads analyte labels from metadata in "Protocol Information" sheet
    try:
        protocol_df = pd.read_excel(
            plate_path, sheet_name='Protocol Information', header=None,
            index_col=0, dtype=str, engine='openpyxl'
        )
    except (ValueError, KeyError, XLRDError):
        raise ValueError(
            '"Protocol Information" sheet not found in {}'.format(plate_path)
        )
    protocol_df_index = [str(x).lower().replace(' ', '')
                         for x in protocol_df.index.tolist()]
    start_row = np.nan
    try:
        start_row = protocol_df_index.index('platelayout')
    except ValueError:
        raise ValueError(
            '"Plate Layout" information not found on Protocol Information sheet'
        )
    label_df = pd.read_excel(
        plate_path, sheet_name='Protocol Information', header=None,
        skiprows=start_row+2, index_col=0, dtype=str, engine='openpyxl'
    ).reset_index(drop=True)
    label_df = trim_dataframe(label_df, plate_path)
    label_df = label_df.replace(
        to_replace='[bB][lL][aA][nN][kK]', value='blank', regex=True
    )
    label_array = copy.deepcopy(label_df).to_numpy()
    # Removes NaN values from label_set
    label_set = []
    for label in label_array.flatten('C'):  # Flattens in row-major
    # style, i.e. np.array([[1, 2], [3, 4]]) => array([1, 2, 3, 4])
        if type(label) == str:
            if not label in label_set:
                label_set.append(label)
        elif type(label) in [int, float]:
            if (not np.isnan(label)) and (not str(label) in label_set):
                label_set.append(str(label))

    # Reads peptide layout and names from metadata in "Protocol Information"
    # sheet
    start_row = np.nan
    try:
        start_row = protocol_df_index.index('peptidelayout')
    except ValueError:
        raise ValueError(
            '"Peptide Layout" information not found on Protocol Information sheet'
        )
    peptide_arrang = pd.read_excel(
        plate_path, sheet_name='Protocol Information', header=None,
        skiprows=start_row+1, index_col=None, dtype=str, engine='openpyxl'
    ).reset_index(drop=True)
    peptide_arrang = trim_dataframe(peptide_arrang, plate_path)
    r_dim = peptide_arrang.shape[0]
    c_dim = peptide_arrang.shape[1]
    plate_peptides = copy.deepcopy(peptide_arrang).to_numpy().flatten('F').tolist()
    # Flattens in column-major style,
    # i.e. np.array([[1, 2], [3, 4]]) => array([1, 3, 2, 4])

    # Checks that peptides listed in the xlsx file are the same as those input
    # by the user, and that there are no repeated peptides
    plate_peptides_no_nan = []
    for peptide in plate_peptides:
        if type(peptide) is float:
            if np.isnan(peptide):
                pass
            else:
                plate_peptides_no_nan.append(peptide)
        else:
            plate_peptides_no_nan.append(peptide)
    peptide_list = []
    for fluorophore, sub_peptide_list in peptide_dict.items():
        if (
                (split == fluorophore)
            and (sorted(sub_peptide_list) == sorted(plate_peptides_no_nan))
        ):
            peptide_list = ['{}_{}'.format(fluorophore, peptide)
                            for peptide in sub_peptide_list]
            break
    if peptide_list == []:
        raise PlateLayoutError('Peptides tested in {} don\'t match peptides'
                               ' specified by user'.format(plate_path))

    if len(set(plate_peptides)) != (r_dim * c_dim):
        multi_peptides = []
        for peptide in copy.deepcopy(plate_peptides):
            if type(peptide) == float:
                if np.isnan(peptide):
                    continue
            if plate_peptides.count(peptide) > 1 and not peptide in multi_peptides:
                multi_peptides.append(peptide)
        if multi_peptides != []:
            raise PlateLayoutError(
                'Peptide(s)\n{}\nlisted more than once in plate layout from '
                '{}'.format(multi_peptides, plate_path)
            )
    # This check shouldn't be needed, since if have already checked that the
    # peptide list provided by the user matches that parsed in from the xlsx
    # file, and that there are no repeated peptides in the list parsed in from
    # the xlsx file, then logically there cannot be any repeated peptides in the
    # user-specified list. However, have included this check anyway just in case
    # the tests above are altered, to ensure that the code continues to check
    # for repeat readings in the user-specified list.
    if len(set(peptide_list)) != len(peptide_list):
        raise PlateLayoutError(
            'One or more peptides specified for split {} listed more than '
            'once:\n{}'.format(fluorophore, peptide_list)
        )

    # Reads fluorescence data from "End point" sheet. Calculates plate
    # dimensions from analyte label and peptide layout dataframes
    try:
        plate_df = pd.read_excel(
            plate_path, sheet_name='End point', header=None, index_col=1,
            dtype=str, engine='openpyxl'
        )
    except (ValueError, KeyError, XLRDError):
        raise ValueError(
            '"End point" sheet not found in {}'.format(plate_path)
        )
    plate_df_index = [str(x).lower().replace(' ', '')
                      for x in plate_df.index.tolist()]
    start_row = np.nan
    for index, row in enumerate(plate_df_index):
        if row.startswith('{}.rawdata'.format(gain)):
            start_row = index
            break
    if np.isnan(start_row):
        raise ValueError(
            'Failed to locate plate in {}\n- expect to be marked as e.g. '
            '"1.rawdata"'.format(plate_path)
        )
    nrows = r_dim*label_df.shape[0]
    ncols = c_dim*label_df.shape[1]
    fluor_df = pd.read_excel(
        plate_path, sheet_name='End point', skiprows=start_row+1, nrows=nrows,
        index_col=None, usecols=range(1, ncols+1), dtype=np.float64,
        engine='openpyxl'
    ).reset_index(drop=True)

    # Combines fluorescence data collected for the same analyte into a dataframe
    grouped_fluor_data = OrderedDict()

    for analyte in label_set:
        grouped_fluor_data[analyte] = []

    for index, analyte in np.ndenumerate(label_array):
        if type(analyte) is float:
            if np.isnan(analyte):
                continue

        index_r = index[0]
        index_c = index[1]

        rows = np.arange((r_dim*index_r), ((r_dim*index_r)+(r_dim+1)))
        # + (r+1) so that row range covers r rows
        columns = np.arange(c_dim*(index_c), ((c_dim*index_c)+(c_dim+1)))
        # + (c+1) so that column range covers c columns

        sub_fluor_df = fluor_df.iloc[
            rows[0]:rows[-1], columns[0]:columns[-1]
        ].reset_index(drop=True)
        sub_fluor_vals = copy.deepcopy(sub_fluor_df).to_numpy().flatten('F')
        # Flattens in column-major style (same as applied to peptide dataframe),
        # i.e. np.array([[1, 2], [3, 4]]) => array([1, 3, 2, 4])

        ordered_fluor_vals = [plate_path, analyte]
        count = len(grouped_fluor_data[analyte])+1
        for fluor_peptide in peptide_list:  # peptide_list (i.e. user-specified
        # list) to ensure same order of features is used to parse each plate,
        # even if the barrels are organised in a different order on different
        # plates
            peptide = fluor_peptide.split('_')[-1]
            r_loc, c_loc = (copy.deepcopy(peptide_arrang).to_numpy() == peptide).nonzero()
            r_loc = r_loc[0]  # Already checked that peptide is present once only above
            c_loc = c_loc[0]  # Already checked that peptide is present once only above
            fluor_reading = sub_fluor_df.iloc[r_loc, c_loc]
            ordered_fluor_vals.append(fluor_reading)
        grouped_fluor_data[analyte].append(pd.DataFrame({count: ordered_fluor_vals}))

    for analyte in label_set:
        analyte_df = pd.concat(
            grouped_fluor_data[analyte], axis=1, ignore_index=True
        ).transpose().reset_index(drop=True)
        df_columns = ['Plate', 'Analyte'] + peptide_list
        analyte_df.columns = df_columns
        if analyte_df.isna().any(axis=None) is np.bool_(True):
            raise ValueError('NaN value detected in {}'.format(plate_path))
        grouped_fluor_data[analyte] = analyte_df

    return fluor_df, grouped_fluor_data, peptide_list


def draw_scatter_plot(
    grouped_fluor_data, results_dir, features, analytes, repeat_id, test=False
):
    """
    Generates strip plot showing variation in repeated data points for each
    dataframe included in grouped_fluor_data dictionary (which is organised into
    analyte_name: analyte_dataframe key: value pairs)

    Input
    ----------
    - grouped_fluor_data: Dictionary of dataframes of fluorescence readings for
    each analyte.
    - results_dir: Path (either absolute or relative) to directory where
    output files should be saved.
    - features: List of peptides to include in the plot
    - analytes: List of analytes to include in the plot
    - repeat_id: Unique identifier of the repeat into which the data has been
    grouped
    - test: Boolean describing whether the function is being run during the
    program's unit tests - by default is set to False
    """

    repeat_id = repeat_id.replace('.xlsx', '')
    print('Drawing scatter plot of data spread for {} in {}'.format(
        repeat_id, results_dir.split('/')[-1]
    ))

    if features == []:
        raise ValueError('List of peptides to include in the plot is empty')
    if analytes == []:
        raise ValueError('List of analytes to include in the plot is empty')

    plot_data = OrderedDict()
    for analyte, analyte_df in copy.deepcopy(grouped_fluor_data).items():
        if analyte in analytes:
            extra_column = pd.DataFrame({'Analyte': [analyte]*analyte_df.shape[0]})
            try:
                analyte_df = pd.concat(
                    [analyte_df[features], extra_column], axis=1
                ).reset_index(drop=True)
                plot_data[analyte] = analyte_df
            except KeyError:
                pass  # If more than one split, will be some dataframes with
                # columns not in [features] list
    if plot_data == OrderedDict():
        raise ValueError(
            'Failed to parse the input dataframes - check that "features" and '
            '"analytes" lists match the features and analytes analysed in one '
            'or more of the dataframes in "grouped_fluor_data"'
        )
    plot_df = pd.concat(plot_data.values(), axis=0).reset_index(drop=True)
    plot_df = pd.melt(
        plot_df, id_vars='Analyte', var_name='Peptide', value_name='Reading'
    ).reset_index(drop=True)

    value_check = plot_df[['Reading']].applymap(lambda x: isinstance(x, (int, float)))
    value_check = value_check.all(axis=None, skipna=False)
    if value_check is np.bool_(False):
        raise ValueError(
            'Non-numeric value found in input dataframe for {}'.format(repeat_id)
        )
    # The error below shouldn't ever be raised as only dataframes whose columns
    # match [features] are incorporated into plot_df, but have left this check
    # in place in case the code above changes to ensure that this check is
    # performed at some stage in the function
    for feature in features:
        if not feature in plot_df['Peptide'].tolist():
            raise ValueError(
                'Peptide {} not present in the input data for '
                '{}'.format(feature, repeat_id)
            )
    for analyte in analytes:
        if not analyte in plot_df['Analyte'].tolist():
            raise ValueError(
                'Analyte {} not present in the input data for '
                '{}'.format(analyte, repeat_id)
            )

    if test is True:
        plot_df.to_pickle('{}/Plot_data.pkl'.format(results_dir))

    # Sorts analytes into order. Assumes that if there are multiple dilutions
    # for a particular analyte, dilution Y is labelled as 'AnalyteX_Y'
    start_analytes = []
    end_analytes = OrderedDict()
    for key in plot_data.keys():
        try:
            start_key = '_'.join(key.split('_')[:-1])
            end_key = key.split('_')[-1]
            if '.' in end_key:
                end_key = float(key.split('_')[-1])
            else:
                end_key = int(key.split('_')[-1])
            if not start_key in end_analytes:
                end_analytes[start_key] = []
            if not end_key in end_analytes[start_key]:
                end_analytes[start_key].append(end_key)
        except ValueError:
            start_analytes.append(key)
    sorted_analytes = []
    sorted_analytes += sorted(start_analytes)
    for key in sorted(end_analytes.keys()):
        val_list = sorted(end_analytes[key])
        for val in val_list:
            sorted_analytes.append('{}_{}'.format(key, val))
    if test is True:
        with open('{}/Sorted_analytes.pkl'.format(results_dir), 'wb') as f:
            pickle.dump((sorted_analytes), f)

    # Draws stripplot of data points
    plt.clf()
    plt.figure(figsize=(15,5))
    sns.stripplot(
        data=plot_df, x='Peptide', y='Reading', hue='Analyte',
        hue_order=sorted_analytes, dodge=True
    )
    plt.xlabel('Peptide')
    plt.ylabel('Reading')
    plt.xticks(rotation=90)
    plt.savefig('{}/{}_data_spread.svg'.format(results_dir, repeat_id))
    if test is False:
        plt.show()


def check_for_saturation(
    plate, plate_path, peptide_list, min_fluor, max_fluor
):
    """
    Checks that no fluorescence readings are outside of the fluorescence range
    that can be measured by the plate reader.

    Input
    ----------
    - plate: Dictionary of dataframes of fluorescence readings from a plate
    - plate_path: File path to the xlsx file from which the plate dictionary
    was parsed
    - peptide_list: List of barrel names
    - min_fluor: Minimum fluoresence reading the plate can measure
    - max_fluor: Maximum fluorescence reading the plate can measure

    Output
    ----------
    - saturated_readings: List of saturated measurements, identified by plate
    file path and barrel name
    """

    saturated_readings = []
    for analyte, orig_fluor_df in plate.items():
        fluor_df = copy.deepcopy(orig_fluor_df)[peptide_list].reset_index(drop=True)
        for index, reading in np.ndenumerate(copy.deepcopy(fluor_df).to_numpy()):
            column = index[1]
            barrel = fluor_df.columns[column]
            if reading <= min_fluor or reading >= max_fluor:
                data_point = [plate_path, analyte, barrel, reading]
                saturated_readings.append(data_point)

    return saturated_readings

def highlight_outliers(
    plate_df, remove_outliers, plate_outliers=None, alpha=0.05, drop_thresh=2,
    user_k_max=np.nan
):
    """
    For each column in an input dataframe, highlights data points identified as
    outliers by a generalised ESD test
    (https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm).

    Input
    ----------
    - plate_df: DataFrame of input fluorescence readings
    - remove_outliers: Boolean, if set to True will discard outliers identified
    via the generalised ESD test.
    - plate_outliers: List of fluoresence readings previously identified as
    outliers. Further outliers identified in plate_df are appended to this list.
    - alpha: Significance level for the generalised ESD test, default 0.05
    - drop_thresh: The minimum number of outlier readings a sample requires to
    be discarded from the dataset (only applied when remove_outliers is set to
    True), default 2
    - user_k_max: The maximum number of runs of the generalised ESD test (and
    hence the maximum number of outliers that can be identified) to be performed.
    By default is set to np.nan, in which case an appropriate value based upon
    the number of readings is automatically selected

    Output
    ----------
    sub_plate_df: DataFrame of input fluorescence readings, with samples
    containing more than drop_thresh outliers removed
    - outlier_dict: Dictionary that records for each row in plate_df (keys) the
    total number plus the identities of any identified outliers
    - plate_outliers: Updated list of outlier fluorescence readings
    """

    if plate_outliers is None:
        plate_outliers = []

    plate_df = plate_df.reset_index(drop=True)
    if plate_df.empty:
        raise ValueError('Empty dataframe:\n{}'.format(plate_df))

    if not 'Analyte' in plate_df.columns:
        raise ValueError('No \'Analyte\' column in input dataframe')

    outlier_dict = OrderedDict()
    for index in range(plate_df.shape[0]):
        outlier_dict[index] = [0, []]

    features = [feature for feature in plate_df.columns
                if not feature in ['Plate', 'Analyte']]
    for feature in features:
        f_index = list(plate_df.columns).index(feature)

        # Checks that none of the features has entirely NaN values
        if set(pd.isnull(plate_df[feature])) == {True}:
            raise ValueError(
                'All values for {} are NaN:\n{}'.format(feature, plate_df)
            )

        feature_array = copy.deepcopy(plate_df[feature]).to_numpy(dtype=np.float64)
        n = feature_array[~np.isnan(feature_array)].shape[0]

        if n <= 0:
            raise ValueError(
                'Empty dataframe for {}:\n{}'.format(feature, feature_array)
            )

        k_max = np.nan
        if not np.isnan(user_k_max):
            k_max = user_k_max
        else:
            if 0 < n <= 2:
                k_max = 0  # Outlier identification not possible if 2 or fewer repeats
            elif 2 < n < 15:
                k_max = 1
            elif 15 <= n < 25:
                k_max = 2
            else:
                k_max = math.floor(n / 2)

        k = 0
        while k < k_max:
            # Calculation of Grubbs test threshold value. Have double-checked
            # that the order of addition and subtraction is correct in these
            # calculations.
            t = stats.t.ppf((1 - (alpha / (2*(n-k)))), (n-k-2))
            numerator = (n-k-1) * t
            denominator = np.sqrt(n-k) * np.sqrt((n-k-2)+np.square(t))
            g_critical = numerator / denominator

            # Calculation of Grubbs test statistic
            abs_diff = np.abs(feature_array - np.nanmean(feature_array))
            max_val = np.nanmax(abs_diff, axis=0)
            max_index = np.nanargmax(abs_diff, axis=0)

            if max_val == 0:  # I.e. If all the readings are the same
                g_calculated = g_critical
            else:
                g_calculated = max_val / np.nanstd(feature_array, ddof=1)  # Uses
            # sample standard deviation as required by test

            if g_calculated > g_critical:
                feature_array[max_index] = np.nan
                analyte = plate_df['Analyte'][max_index]
                plate_outliers.append(
                    [analyte, feature, plate_df[feature][max_index]]
                )
                if remove_outliers is True:
                    outlier_dict[max_index][0] += 1
                    outlier_dict[max_index][1].append(f_index)

                k += 1

            else:
                break

    if remove_outliers is True:
        keep_indices = [index for index in range(plate_df.shape[0])
                        if outlier_dict[index][0] < drop_thresh]
        sub_plate_df = copy.deepcopy(plate_df).iloc[keep_indices].reset_index(drop=True)
        return sub_plate_df, outlier_dict

    elif remove_outliers is False:
        return plate_outliers


def calc_median(plate_df, raise_warning):
    """
    For each column in an input dataframe, calculates the median value.

    Input
    ----------
    - plate_df: DataFrame of input fluorescence readings
    - raise_warning: Boolean that raises an error if a median value is
    calculated to be NaN

    Output
    ----------
    - median_df: DataFrame of the median fluorescence value of each column of
    fluoresence readings in the input dataframe. Outliers are not excluded
    before the median calculation, however the median value should be relatively
    robust to outliers
    """

    drop_columns = ['Plate', 'Analyte']
    features = [feature for feature in plate_df.columns
                if feature not in drop_columns]

    median_df = plate_df[features].reset_index(drop=True)
    median_df = median_df.median(axis=0, skipna=True).to_frame().transpose()
    # Transposing ensures features are the columns rather than the rows in median_df

    # Raises error if any NaN values in median_df
    if raise_warning is True:
        if median_df.isnull().any().any() == True:
            nan_features = []
            for index, val in enumerate(median_df.isnull().any()):
                if val is True:
                    nan_features.append(median_df.columns[index])
            raise NaNFluorescenceError(
                '\x1b[31m ERROR - median reading/s for {} is/are calculated to '
                'be NaN. \033[0m'.format(nan_features)
            )

    return median_df


def scale_min_max(
    scale_method, plate, plate_name, no_pep, split_name, cols_ignore,
    plate_outliers, outlier_excl_thresh=0.05, drop_thresh=2, k_max=np.nan,
    test=False
):
    """
    Scales data on plate between min fluorescence reading and max fluorescence
    reading (the values of the min and max readings are dependent upon the
    scaling method selected)

    Input
    ----------
    - scale_method: Either 'fluorophore' or 'analyte_fluorophore'. If set to
    'fluorophore', the min fluorescence reading used in the scaling calculation
    is the fluorophore on its own, whereas if set to 'analyte_fluorophore', the
    min fluoresence reading used is the fluorescence of the analyte with the
    fluorophore. In both cases, the max fluorescence reading is the peptide with
    the fluorophore.
    - plate: Dictionary of DataFrames of fluorescence data parsed from a single
    plate (as is output from parse_xlsx_to_dataframe), where the keys are the
    analytes tested on the plate and the values are the corresponding DataFrames
    of fluorescence readings
    - plate_name: Plate ID
    - no_pep: Dictionary of the name given to the sample without peptide in
    input peptide array (value) for each split (key)
    - split_name: Name of split, which has been added as a prefix to all of the
    peptide names in the plate columns
    - cols_ignore: List of names of peptides to exclude from the analysis
    - plate_outliers: List of fluoresence readings previously identified as
    outliers, to which outliers in the current plate will be appended
    - outlier_excl_thresh: Significance level for discarding outliers (via a
    generalised ESD test), default value is 0.05
    - drop_thresh: The minimum number of outlier readings a sample requires
    to be discarded from the dataset (only applied when remove_outliers is
    set to True), default value is 2
    - k_max: The maximum number of outliers to exclude with a generalised ESD
    test. If leave as default np.nan, an appropriate threshold will be selected
    based upon the size of the dataset.
    - test: Boolean describing whether the function is being run during the
    program's unit tests - by default is set to False

    Output
    ----------
    - scaled_plate: Dictionary of dataframes of min-max scaled fluorescence
    readings for each analyte
    - plate_outliers: List of outlier fluoresence readings updated now to
    contain any outliers in the input dataframe of fluorescence readings
    (plate_df)
    - peptide_list: List of peptides included on the input plate
    """

    try:
        blank_data = copy.deepcopy(plate['blank'])
    except KeyError:
        raise PlateLayoutError('No blank readings (= peptide + fluorophore '
                               'without analyte) included on plate')

    if not scale_method in ['fluorophore', 'analyte_fluorophore']:
        raise ValueError(
            'Expect argument "scale_method" to be set to either '
            '"fluorophore" or "analyte_fluorophore", not '
            '{}'.format(scale_method)
        )

    blank_cols = [col.lower().replace(' ', '') for col in blank_data.columns]
    if not 'analyte' in blank_cols:
        raise ValueError(
            '\'Analyte\' column not in input dataframe:\n{}'.format(blank_data)
        )
    if not 'plate' in blank_cols:
        raise ValueError(
            '\'Plate\' column not in input dataframe:\n{}'.format(blank_data)
        )

    if blank_data.drop(['Analyte', 'Plate'], axis=1).isna().any(axis=None) is np.bool_(True):
        raise ValueError(
            'NaN reading on plate {} for blank readings'.format(plate_name)
        )

    plate_outliers = highlight_outliers(
        blank_data, remove_outliers=False, plate_outliers=plate_outliers,
        alpha=outlier_excl_thresh, drop_thresh=drop_thresh,
        user_k_max=k_max
    )
    blank_data = calc_median(blank_data, raise_warning=True)

    no_pep_str = '{}_{}'.format(split_name, no_pep[split_name])
    scaled_plate = OrderedDict()
    analytes = [analyte for analyte in list(plate.keys()) if analyte != 'blank']
    for analyte in analytes:
        fluor_data = copy.deepcopy(plate[analyte]).reset_index(drop=True)

        # Checks that no peptide blank is available for each analyte
        if not no_pep_str in fluor_data.columns:
            raise PlateLayoutError(
                'No reading for {} for analyte {} on plate {}.\nNo peptide '
                'blank is required for min max scaling'.format(
                    no_pep_str, analyte, plate_name
                )
            )

        plate_cols = [col.lower().replace(' ', '') for col in fluor_data.columns]
        if not 'analyte' in plate_cols:
            raise ValueError(
                '\'Analyte\' column not in input dataframe:\n{}'.format(fluor_data)
            )
        if not 'plate' in plate_cols:
            raise ValueError(
                '\'Plate\' column not in input dataframe:\n{}'.format(fluor_data)
            )

        if fluor_data.drop(['Analyte', 'Plate'], axis=1).isna().any(axis=None) is np.bool_(True):
            raise ValueError(
                'NaN reading on plate {} for {} readings'.format(plate_name, analyte)
            )

        # Checks that fluorescence of analyte + DPH is lower than all analyte +
        # DPH + peptide combinations (except for peptides in cols_ignore). Won't
        # work for any NaN median values, but these will be dealt with at a
        # later stage (when readings from the same repeat are combined across
        # plates)
        plate_outliers = highlight_outliers(
            fluor_data, remove_outliers=False, plate_outliers=plate_outliers,
            alpha=outlier_excl_thresh, drop_thresh=drop_thresh,
            user_k_max=k_max
        )
        median_fluor_data = calc_median(fluor_data, raise_warning=True)
        min_fluor_analyte = median_fluor_data[no_pep_str][0]
        for peptide in list(median_fluor_data.columns):
            if peptide != no_pep_str and not peptide in cols_ignore:
                fluor_analyte_pep = median_fluor_data[peptide][0]
                if fluor_analyte_pep > min_fluor_analyte:
                    continue
                else:
                    if test is True:
                        raise ValueError(
                            'Median fluorescence of ({} + {} + fluorophore) is '
                            'less than fluorescence of ({} + {} + fluorophore) '
                            'alone on plate {}'.format(
                                analyte, peptide, analyte, no_pep_str, plate_name
                            )
                        )
                    else:
                        print('\x1b[31m WARNING - median fluorescence of ({} + '
                              '{} + fluorophore) is less than fluorescence of '
                              '({} + fluorophore) alone on plate {}. Analysis '
                              'will continue but please CHECK YOUR DATA. \033[0m'
                              ''.format(analyte, peptide, analyte, plate_name))

        # Performs min max scaling for each feature
        for index_c, peptide in enumerate(list(fluor_data.columns)):
            if (
                    not peptide in ['Plate', 'Analyte', no_pep_str]
                and not peptide in cols_ignore
            ):
                # Applies min max scaling to each reading
                max_fluor = blank_data[peptide][0]
                min_fluor_demoninator = blank_data[no_pep_str][0]
                if scale_method == 'fluorophore':
                    min_fluor_numerator = blank_data[no_pep_str][0]
                elif scale_method == 'analyte_fluorophore':
                    min_fluor_numerator = median_fluor_data[no_pep_str][0]

                for index_r, reading in enumerate(fluor_data[peptide].tolist()):
                    if (max_fluor-min_fluor_demoninator) > 0:
                        scaled_val = (  (reading-min_fluor_numerator)
                                      / (max_fluor-min_fluor_demoninator))
                    else:
                        scaled_val = np.nan
                        if max_fluor == min_fluor_demoninator:
                            raise MinMaxFluorescenceError(
                                'Min and max fluorescence readings for peptide '
                                '{} on plate {} are the same'.format(
                                    peptide, fluor_data['Plate'][index_r]
                                )
                            )
                        elif max_fluor < min_fluor_demoninator:
                            raise MinMaxFluorescenceError(
                                'Median max. fluorescence reading for peptide '
                                '{} on plate {} is smaller than the '
                                'corresponding median min. fluorescence '
                                'reading'.format(
                                    peptide, fluor_data['Plate'][index_r]
                                )
                            )
                    fluor_data.iloc[index_r, index_c] = scaled_val

        drop_columns = [no_pep_str] + cols_ignore
        for col in drop_columns:
            try:
                fluor_data = fluor_data.drop([col], axis=1)
            except KeyError:
                continue  # As barrels can be split across different plates,
                # user-specified peptides to ignore across all splits may not be
                # found in an individual split
        peptide_list = [col for col in fluor_data.columns
                        if not col in ['Plate', 'Analyte']]

        scaled_plate[analyte] = fluor_data

    return scaled_plate, plate_outliers, peptide_list


def draw_heatmap_fingerprints(
    fluor_df, results_dir, scale, class_order=None, prefix='', test=False
):
    """
    Draws a heatmap for each analyte representing the median reading for each
    peptide

    Input
    ----------
    - fluor_df: Dataframe of fluorescence readings output from running
    combine_plate_readings
    - results_dir: Path (either absolute or relative) to directory where
    output files will be saved. This directory must already exist.
    - scale: Boolean, specifies whether to scale the input dataframe of
    fluorescence readings and compare it to the original (unscaled) data
    - class_order: List of analytes. Provide this list if you want the graphs to
    be drawn in a particular order, otherwise this function will automatically
    determine the analytes present (and plot them in the order obtained using
    the sorted() function)
    - prefix: Prefix to add to the file names of saved plots
    - test: Boolean describing whether the function is being run during the
    program's unit tests - by default is set to False

    Output
    ----------
    test_results: Dictionary of the dataframes used to create each plot,
    returned if test is set to True
    """

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            'Directory {} does not exist'.format(results_dir)
        )
    if os.path.isdir('{}/Heatmap_plots'.format(results_dir)):
        raise FileExistsError(
            'Directory {}/Heatmap_plots exists - please rename this directory if'
            ' you wish to run this function to avoid overwriting the original '
            'directory'.format(results_dir)
        )
    os.mkdir('{}/Heatmap_plots'.format(results_dir))

    if not 'Analyte' in fluor_df.columns:
        raise KeyError(
            'No "Analyte" column in input dataframe'
        )

    classes = sorted([analyte for analyte in set(fluor_df['Analyte'].tolist())])
    if not class_order is None:
        try:
            if classes != sorted(class_order):
                raise ValueError(
                    'Expect class_order to be a list of analytes including every '
                    'analyte in the dataset once. Instead is set to:\n'
                    '{}'.format(class_order)
                )
            else:
                classes = class_order
        except TypeError:
            raise ValueError(
                'Expect class_order to be a list of analytes including every '
                'analyte in the dataset once. Instead is set to:\n'
                '{}'.format(class_order)
            )

    if not isinstance(scale, bool):
        raise ValueError(
            'Expect Boolean (True/False) input for whether or not to scale the '
            'data, instead have {}, type {}'.format(scale, type(scale))
        )

    if fluor_df.isna().any(axis=None, skipna=False) is np.bool_(True):
        raise ValueError(
            'NaN reading(s) found in input dataframe:\n{}'.format(fluor_df)
        )

    sub_fluor_df = copy.deepcopy(fluor_df).drop('Analyte', axis=1)
    if sub_fluor_df.applymap(
        lambda x: isinstance(x, (int, float))).all(axis=None, skipna=False
    ) is np.bool_(False):
        raise ValueError(
            'Non-numeric value in input dataframe - expect all values in input'
            ' dataframe to be integers / floats:\n{}'.format(sub_fluor_df)
        )

    if np.isinf(sub_fluor_df.values).any(axis=None) is np.bool_(True):
        raise ValueError(
            'Infinite reading(s) found in input dataframe:\n'.format(fluor_df)
        )

    cols = [col for col in fluor_df.columns if col != 'Analyte']
    df_dict = OrderedDict({'Original_data': copy.deepcopy(fluor_df)})
    if scale is True:
        scaled_df = RobustScaler().fit_transform(
            copy.deepcopy(fluor_df)[cols]
        )
        scaled_df = pd.DataFrame(scaled_df, columns=cols)
        scaled_df = pd.concat([scaled_df, fluor_df[['Analyte']]], axis=1)
        df_dict['Scaled_data'] = scaled_df

    test_results = OrderedDict({'All_data': df_dict})
    for label, df in df_dict.items():
        print('\n\n\n{}:\n'.format(label))

        df_x_val = df.drop(['Analyte'], axis=1)
        class_median_x_val = OrderedDict()
        class_median_x_val['min'] = np.nan
        class_median_x_val['max'] = np.nan
        for index, class_name in enumerate(classes):
            class_indices = [
                n for n in range(df.shape[0]) if df['Analyte'][n] == class_name
            ]
            class_df = df_x_val.values[class_indices, :]
            median_class_df = np.median(class_df, axis=0)
            class_median_x_val[class_name] = median_class_df

            if index == 0:
                class_median_x_val['min'] = np.amin(median_class_df)
                class_median_x_val['max'] = np.amax(median_class_df)
            else:
                if np.amin(median_class_df) < class_median_x_val['min']:
                    class_median_x_val['min'] = np.amin(median_class_df)
                if np.amax(median_class_df) > class_median_x_val['max']:
                    class_median_x_val['max'] = np.amax(median_class_df)

        test_results[label] = class_median_x_val
        for class_name in classes:
            print('\n{}'.format(class_name))

            sns.set(rc={'figure.figsize':(23,0.4)})
            plt.clf()
            sns.heatmap(
                np.array([class_median_x_val[class_name]]),
                vmin=class_median_x_val['min'], vmax=class_median_x_val['max'],
                annot=False, cmap='RdBu_r', cbar=True,
                xticklabels=df_x_val.columns, yticklabels=False, linecolor='k',
                linewidths=2
            )
            plt.savefig(
                '{}/Heatmap_plots/{}{}_{}_median_barrel_readings_heatmap.svg'.format(
                    results_dir, prefix, label, class_name
                )
            )
            if test is False:
                plt.show()

    if test is True:
        return test_results


def draw_boxplots(
    fluor_df, results_dir, scale, cushion, class_order=None, prefix='',
    test=False
):
    """
    Draw a boxplot representing the range of readings for each analyte

    Input
    ----------
    - fluor_df: Dataframe of fluorescence readings output from running
    combine_plate_readings
    - results_dir: Path (either absolute or relative) to directory where
    output files will be saved. This directory must already exist.
    - scale: Boolean, specifies whether to scale the input dataframe of
    fluorescence readings and compare it to the original (unscaled) data
    - cushion: Size of buffer to add to the y-axes, whose range is between
    (smallest value in dataset - cushion, largest value in dataset + cushion)
    - class_order: List of analytes. Provide this list if you want the graphs to
    be drawn in a particular order, otherwise this function will automatically
    determine the analytes present (and plot them in the order obtained using
    the sorted() function)
    - prefix: Prefix to add to the file names of saved plots
    - test: Boolean describing whether the function is being run during the
    program's unit tests - by default is set to False

    Output
    ----------
    test_results: Dictionary of the dataframes used to create each plot,
    returned if test is set to True
    """

    # Tests that input arguments are suitable for running the function
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            'Directory {} does not exist'.format(results_dir)
        )

    if os.path.isdir('{}/Boxplots'.format(results_dir)):
        raise FileExistsError(
            'Directory {}/Boxplots exists - please rename this directory if you'
            ' wish to run this function to avoid overwriting the original '
            'directory'.format(results_dir)
        )
    os.mkdir('{}/Boxplots'.format(results_dir))

    if not 'Analyte' in fluor_df.columns:
        raise KeyError(
            'No "Analyte" column in input dataframe'
        )

    classes = sorted([analyte for analyte in set(fluor_df['Analyte'].tolist())])
    if not class_order is None:
        try:
            if classes != sorted(class_order):
                raise ValueError(
                    'Expect class_order to be a list of analytes including every '
                    'analyte in the dataset once. Instead is set to:\n'
                    '{}'.format(class_order)
                )
            else:
                classes = class_order
        except TypeError:
            raise ValueError(
                'Expect class_order to be a list of analytes including every '
                'analyte in the dataset once. Instead is set to:\n'
                '{}'.format(class_order)
            )

    if not isinstance(cushion, (int, float)):
        raise ValueError(
            'Unexpected value for cushion: {}\nExpect an integer or a '
            'float.'.format(cushion)
        )

    if not isinstance(scale, bool):
        raise ValueError(
            'Expect Boolean (True/False) input for whether or not to scale the '
            'data, instead have {}, type {}'.format(scale, type(scale))
        )

    if fluor_df.isna().any(axis=None, skipna=False) is np.bool_(True):
        raise ValueError(
            'NaN reading(s) found in input dataframe:\n{}'.format(fluor_df)
        )

    sub_fluor_df = copy.deepcopy(fluor_df).drop('Analyte', axis=1)
    if sub_fluor_df.applymap(
        lambda x: isinstance(x, (int, float))).all(axis=None, skipna=False
    ) is np.bool_(False):
        raise ValueError(
            'Non-numeric value in input dataframe - expect all values in input'
            ' dataframe to be integers / floats:\n{}'.format(sub_fluor_df)
        )

    if np.isinf(sub_fluor_df.values).any(axis=None) is np.bool_(True):
        raise ValueError(
            'Infinite reading(s) found in input dataframe:\n{}'.format(fluor_df)
        )

    # Generates plots
    cols = [col for col in fluor_df.columns if col != 'Analyte']
    df_dict = OrderedDict({'Original_data': copy.deepcopy(fluor_df)})
    if scale is True:
        scaled_df = RobustScaler().fit_transform(
            copy.deepcopy(fluor_df)[cols]
        )
        scaled_df = pd.DataFrame(scaled_df, columns=cols)
        scaled_df = pd.concat([scaled_df, fluor_df[['Analyte']]], axis=1)
        df_dict['Scaled_data'] = scaled_df

    test_results = OrderedDict({'All_data': df_dict})
    for label, df in df_dict.items():
        print('\n\n\n{}:\n'.format(label))

        ymin = df.drop('Analyte', axis=1).min().min() - cushion
        ymax = df.drop('Analyte', axis=1).max().max() + cushion

        melt_df = pd.melt(
            copy.deepcopy(df), id_vars='Analyte', var_name='Barrel',
            value_name='Reading'
        )
        test_results[label] = [ymin, ymax, melt_df]

        sns.set(rc={'figure.figsize':(15,6)})
        plt.clf()
        plt.xticks(rotation=90)
        plt.ylim(ymin, ymax)
        sns.boxplot(
            data=melt_df, x='Barrel', y='Reading', hue='Analyte',
            hue_order=classes
        )
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig('{}/Boxplots/{}{}_barrel_readings_boxplot.svg'.format(
            results_dir, prefix, label
        ))
        if test is False:
            plt.show()

        for class_name in classes:
            print(class_name)
            indices = [int(i) for i in range(df.shape[0])
                       if df['Analyte'][i] == class_name]

            sns.set(rc={'figure.figsize':(15,6)})
            plt.clf()
            plt.xticks(rotation=90)
            plt.ylim(ymin, ymax)
            sns.boxplot(data=df.iloc[indices])
            sns.swarmplot(
                data=df.iloc[indices], size=3, edgecolor='k', linewidth=0.8
            )
            plt.savefig('{}/Boxplots/{}{}_{}_barrel_readings_boxplot.svg'.format(
                results_dir, prefix, label, class_name
            ))
            if test is False:
                plt.show()

    if test is True:
        return test_results


class DefData():

    def __init__(self, results_dir, test=False):
        """
        - results_dir: Path (either absolute or relative) to directory where
        output files should be saved. This directory will be created by the
        program and so should not already exist.
        """

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        else:
            if test is True:
                raise FileExistsError('Directory {} already found in {}'.format(
                    results_dir, os.getcwd()
                ))
            else:
                print('Directory {} already found in {}'.format(
                    results_dir, os.getcwd()
                ))
                remove = ''
                while not isinstance(remove, bool):
                    remove = input('Overwrite {}?\n'.format(results_dir))
                    if remove.strip().lower() in ['true', 'yes', 'y']:
                        remove = True
                    elif remove.strip().lower() in ['false', 'no', 'n']:
                        remove = False
                    else:
                        print('Input not recognised - please specify "yes" or '
                              '"no"')
                        remove = ''
                if remove is True:
                    shutil.rmtree(results_dir)
                    os.mkdir(results_dir)
                else:
                    print('Please specify a unique directory name for results '
                          'to be stored in')
        self.results_dir = results_dir.rstrip('/')


class ParseArrayData(DefData):

    def __init__(
        self, data_dirs_dict, results_dir, repeat_labels_list, peptide_dict,
        control_peptides, control_analytes, gain=1, min_fluor=0,
        max_fluor=260000, test=False
    ):
        """
        - data_dirs_dict: Dictionary of split names (keys) and paths (either
        absolute or relative) to directory containing xlsx files of fluorescence
        readings for that split (values)
        - results_dir: Path (either absolute or relative) to directory where
        output files should be saved. This directory will be created by the
        program and so should not already exist.
        - repeat_labels_list: List of repeat labels used in the xlsx file names
        in data_dir. Xlsx files in this directory should be named such that the
        repeat label is at the end of the file, e.g. ****_repeat_1.xlsx or
        equivalent.
        - peptide_dict: Dictionary of split labels (keys) and the peptides
        included in each split (values), e.g. {'DPH': ['CCHex', 'CCHept'], 'NR':
        ['CCHex', 'CCHex2']}
        - control_peptides: Dictionary of peptides (not including "No Peptide")
        in each split to be excluded from the analysis. E.g. Typically we
        exclude the collapsed barrel control (it is only included to check that
        the fluorescence readings collected from the plate look reasonable),
        hence (for an analysis with a single (DPH) split)
        control_peptides={'DPH': ['GRP35']}.
        - control_analytes: List of analytes (across all splits) to be excluded
        from the analysis. Useful if some analytes have not been included in all
        replicates and/or across all features in the splits.
        - gain: Which dataframe of fluorescence data (collected at different
        gains) to use, default is first listed in xlsx file (gain=1)
        - min_fluor: Minimum fluorescence reading that can be measured by the
        plate reader, default is 0
        - max_fluor: Maximum fluorescence reading that can be measured by the
        plate reader, default is 260,000
        - test: Boolean describing whether the class is being run during the
        program's unit tests - by default is set to False
        """

        DefData.__init__(self, results_dir)

        # Defines directory where files of fluorescence readings are saved
        self.data_dirs_dict = data_dirs_dict
        for data_dir in self.data_dirs_dict.values():
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(
                    'Path to data directory not recognised:\n{}'.format(data_dir)
                )

        # Defines dictionary of lists of file names grouped by repeat label
        self.repeat_list = repeat_labels_list
        if type(self.repeat_list) != list:
            raise TypeError(
                'Expected list of repeat labels, instead got '
                '{}'.format(type(self.repeat_list))
            )

        # Defines dictionary of lists of peptide names grouped by split label
        self.peptide_dict = peptide_dict
        if not type(self.peptide_dict) in [dict, OrderedDict]:
            raise TypeError(
                'Expected dictionary of list of peptide names grouped by split '
                'label, instead got {}'.format(type(self.peptide_dict))
            )
        for fluorophore, peptide_list in self.peptide_dict.items():
            if len(peptide_list) != len(set(peptide_list)):
                multi_peptides = []
                for peptide in copy.deepcopy(peptide_list):
                    if peptide_list.count(peptide) > 1 and not peptide in multi_peptides:
                        multi_peptides.append(peptide)
                raise PlateLayoutError(
                    'Barrel(s) listed more than once in split {}: {}'.format(
                        fluorophore, multi_peptides
                    )
                )
            if '_' in fluorophore:
                raise ValueError(
                    'Neither split nor peptide names should contain the '
                    'character "_"'
                )
            for peptide in peptide_list:
                if '_' in peptide:
                    raise ValueError(
                        'Neither split nor peptide names should contain the '
                        'character "_"'
                    )

        # Makes list of all peptides (across all splits)
        all_peptide_list = []
        for fluorophore, peptide_list in self.peptide_dict.items():
            all_peptide_list += ['{}_{}'.format(fluorophore, peptide_list[n])
                                 for n in range(len(peptide_list))]
        if len(set(all_peptide_list)) != len(all_peptide_list):
            raise Exception(
                'Duplicate peptides detected across splits:\n'
                '{}'.format(all_peptide_list)
            )
        self.peptide_list = all_peptide_list

        # Defines dictionary of lists of control peptides (used to assess the
        # range of readings on the plate to make sure they look sensible, but
        # not used in the analysis) grouped by split label
        self.control_peptides = control_peptides
        if not type(self.control_peptides) in [dict, OrderedDict]:
            raise TypeError(
                'Expected dictionary of list of control peptide names grouped '
                'by split label, instead got {}'.format(type(self.control_peptides))
            )
        for split in self.peptide_dict.keys():
            if not split in self.control_peptides:
                self.control_peptides[split] = []
        for fluorophore, peptide_list in self.control_peptides.items():
            peptide_list = ['{}_{}'.format(fluorophore, peptide)
                            for peptide in peptide_list]
            missing_peptides = [peptide for peptide in peptide_list
                                if not peptide in self.peptide_list]
            if len(missing_peptides) >= 1:
                raise NameError(
                    'Peptide(s):\n{}\nslated for exclusion are not included in list'
                    ' of all peptides'.format(missing_peptides)
                )

        # Defines dictionary of lists of control analytes grouped by split label
        self.control_analytes = control_analytes  # Checks that control analytes
        # exist are later (after fluorescence data has been read in from the
        # xlsx files)
        if type(self.control_analytes) != list:
            raise TypeError(
                'Expected list of control analyte names, instead got '
                '{}'.format(type(self.control_analytes))
            )

        # Defines which set of plate readings to parse (if there are multiple
        # sets of plate readings in the same file), plus the minimum and maximum
        # expected fluorescence readings to allow detection of anomalous and/or
        # saturated readings
        self.gain = gain
        self.min_fluor = min_fluor
        self.max_fluor = max_fluor
        for prop_name, prop_val in {'gain': self.gain,
                                    'min_fluor': self.min_fluor,
                                    'max_fluor': self.max_fluor}.items():
            error = False
            if type(prop_val) != int:
                error = True
            else:
                if prop_name == 'gain':
                    if prop_val < 1:
                        error = True
                else:
                    if prop_val < 0:
                        error = True
            if error is True:
                raise TypeError(
                    'Expected positive integer for {}, instead got '
                    '{}'.format(prop_name, prop_val)
                )
        # Below must be run AFTER checks that min_fluor and max_fluor values are
        # integers
        if self.min_fluor >= self.max_fluor:
            raise ValueError(
                'Expected minimum expected fluorescence reading to be smaller '
                'than maximum expected fluoresence reading, instead values are '
                '{} (min) and {} (max)'.format(self.min_fluor, self.max_fluor)
            )

        # Defines whether the class is being called during the progran's unit
        # tests
        self.test = test

    def group_xlsx_repeats(self, ignore_files=None):
        """
        Groups xlsx files in the directories in self.data_dirs_dict into
        independent repeats (via the name of the repeat included at the end of
        the file name)

        Input
        ----------
        - ignore_files: Dictionary of list of files (value) in each split (key)
        not to be included in the analysis
        """

        if ignore_files is None:
            ignore_files = OrderedDict()

        xlsx_file_dict = OrderedDict()

        # Checks that all files slated for removal a) actually exist and b) are
        # xlsx files
        for split, data_dir in self.data_dirs_dict.items():
            all_files = sorted(os.listdir(data_dir))
            if split in ignore_files.keys():
                for xlsx in ignore_files[split]:
                    xlsx = xlsx.split('/')[-1].split('\\')[-1]
                    if not xlsx in all_files:
                        raise FileNotFoundError(
                            '{} not found in\n{}\n- make sure that all files '
                            'specified to be exluded from the analysis '
                            'exist'.format(xlsx, all_files)
                        )
                    if not xlsx.endswith('.xlsx'):
                        raise ValueError(
                            'File {} specified to be excluded from the analysis'
                            ' is not an xlsx file'.format(xlsx)
                        )
            else:
                ignore_files[split] = []

        for repeat in self.repeat_list:
            xlsx_file_dict[repeat] = OrderedDict()

            for split, data_dir in self.data_dirs_dict.items():
                all_files = sorted(os.listdir(data_dir))
                xlsx_file_dict[repeat][split] = []

                for file_name in all_files:
                    if file_name.endswith('{}.xlsx'.format(repeat)):
                        if (
                                not file_name.startswith('~$')
                            and not file_name in ignore_files[split]
                        ):
                            xlsx_file_dict[repeat][split].append(file_name)
                        elif file_name in ignore_files[split]:
                            print('Successfully removed {}'.format(file_name))

        self.repeat_dict = xlsx_file_dict

    def xlsx_to_scaled_df(
        self, no_pep, scale_method, draw_plot, plot_dir_name='',
        outlier_excl_thresh=0.05, drop_thresh=2, k_max=np.nan
    ):
        """
        Parses grouped xlsx files into dataframes and performs min max scaling.

        Input
        ----------
        - no_pep: Dictionary of name of sample without peptide in peptide array
        (value) for each split (key)
        - scale_method: Method to use for min-max scaling of the data, set to
        either 'fluorophore' or 'analyte_fluorophore'
        - draw_plot: Boolean, determines whether or not to draw a scatter plot
        for each plate, showing the distribution of repeat readings for each
        analyte on that plate (beware, plot drawing is SLOW!)
        - plot_dir_name: Prefix to give the directories created to store the
        generated scatter plots. Only need to set this value if draw_plot is
        True.
        - outlier_excl_thresh: Significance level for discarding outliers (via a
        generalised ESD test), default value is 0.05
        - drop_thresh: The minimum number of outlier readings a sample requires
        to be discarded from the dataset (only applied when remove_outliers is
        set to True), default value is 2
        - k_max: The maximum number of outliers to exclude with a generalised ESD
        test. If leave as default np.nan, an appropriate threshold will be selected
        based upon the size of the dataset.
        """

        error = False
        if type(scale_method) != str:
            error = True
        else:
            scale_method = scale_method.lower().replace(' ', '')
            if not scale_method in ['fluorophore', 'analyte_fluorophore']:
                error = True
        if error is True:
            raise ValueError(
                'Expect argument "scale_method" to be set to either '
                '"fluorophore" or "analyte_fluorophore", not '
                '{}'.format(scale_method)
            )
        if type(draw_plot) != bool:
            raise TypeError('Expect argument "draw_plot" to be a Boolean (True '
                            'or False)')

        if draw_plot is True:
            plot_dirs = [
                '{}/{}_unscaled_data'.format(self.results_dir, plot_dir_name),
                '{}/{}_scaled_data'.format(self.results_dir, plot_dir_name)
            ]
            for plot_dir in plot_dirs:
                if os.path.isdir(plot_dir):
                    raise FileExistsError(
                        'Directory {} already exists'.format(plot_dir)
                    )
                try:
                    os.mkdir(plot_dir)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        'Directory {} doesn\'t exist'.format(
                            '/'.join(self.results_dir.split('/')[:-1])
                        )
                    )

        raw_plate_data = OrderedDict()
        unscaled_data = OrderedDict()
        for repeat in self.repeat_dict.keys():
            for split, xlsx_list in self.repeat_dict[repeat].items():
                for xlsx in xlsx_list:
                    # Converts xlsx spreadsheet into dataframes
                    plate_path = self.data_dirs_dict[split] + xlsx
                    raw_data, plate, peptide_list = parse_xlsx_to_dataframe(
                        plate_path, split, self.peptide_dict, self.gain
                    )
                    raw_plate_data[plate_path] = raw_data
                    unscaled_data[plate_path] = plate

                    # Draws plots to show variation in repeat readings
                    if draw_plot is True:
                        plot_analytes = [
                            analyte for analyte in plate.keys()
                            if not analyte in self.control_analytes
                        ]
                        plot_dir = '{}/{}_unscaled_data'.format(
                            self.results_dir, plot_dir_name
                        )
                        draw_scatter_plot(
                            plate, plot_dir, peptide_list, plot_analytes, xlsx,
                            self.test
                        )

                    # Checks for saturated readings
                    saturated_readings = check_for_saturation(
                        plate, plate_path, peptide_list, self.min_fluor,
                        self.max_fluor
                    )
                    if saturated_readings != []:
                        merged_saturated_readings = '\n'.join([
                            '{} {} {}'.format(x[1], x[2], x[3])
                            for x in saturated_readings
                        ])
                        raise FluorescenceSaturationError(
                            'The following data points appear to have saturated the'
                            ' plate reader on plate {}:\n{}\nTo proceed with '
                            'analysis you need to either remove the identified '
                            'barrels, or replace the fluorescence\nreadings for '
                            'this barrel with readings collected at a different '
                            'gain, for ALL xlsx files in the dataset.'.format(
                                plate_path, merged_saturated_readings
                            )
                        )

        scaled_data = OrderedDict()
        outliers = OrderedDict()
        for repeat in self.repeat_dict.keys():
            scaled_data[repeat] = OrderedDict()
            outliers[repeat] = OrderedDict()

            for split, xlsx_list in self.repeat_dict[repeat].items():
                scaled_data[repeat][split] = OrderedDict()
                outliers[repeat][split] = OrderedDict()

                for xlsx in xlsx_list:
                    plate_path = self.data_dirs_dict[split] + xlsx
                    plate = unscaled_data[plate_path]

                    # Performs min max scaling to enable samples across plates to be
                    # compared and/or combined, plus removes features the user has
                    # specified to ignore.
                    cols_ignore = ['{}_{}'.format(split, peptide)
                                   for peptide in self.control_peptides[split]]
                    outliers[repeat][split][plate_path] = []
                    (
                        scaled_plate, outliers[repeat][split][plate_path],
                        peptide_list
                    ) = scale_min_max(
                        scale_method, plate, plate_path, no_pep, split,
                        cols_ignore, outliers[repeat][split][plate_path],
                        outlier_excl_thresh, drop_thresh, k_max
                    )
                    scaled_data[repeat][split][plate_path] = scaled_plate

                    # Draws plots to show variation in repeat readings
                    if draw_plot is True:
                        plot_analytes = [
                            analyte for analyte in scaled_plate.keys()
                            if not analyte in self.control_analytes
                        ]
                        plot_dir = '{}/{}_scaled_data'.format(
                            self.results_dir, plot_dir_name
                        )
                        draw_scatter_plot(
                            scaled_plate, plot_dir, peptide_list, plot_analytes,
                            xlsx, self.test
                        )

        # Checks that there is data for the same set of analytes in each repeat
        # and across all splits, not considering analytes that the user has
        # specified to be discarded
        count = 0
        orig_analytes = []  # Generates a list of all unique analytes in the
        # dataset
        repeat_1_analytes = OrderedDict()
        for repeat in scaled_data.keys():
            repeat_analytes = OrderedDict()
            count += 1

            for split in scaled_data[repeat].keys():
                repeat_analytes[split] = []

                for plate_path in scaled_data[repeat][split].keys():
                    plate_analytes = [
                        analyte for analyte in
                        scaled_data[repeat][split][plate_path].keys()
                        if not analyte in self.control_analytes
                    ]
                    repeat_analytes[split] += plate_analytes

                repeat_analytes[split] = set(sorted(repeat_analytes[split]))
                if count == 1:
                    repeat_1 = copy.deepcopy(repeat)
                    repeat_1_analytes[split] = copy.deepcopy(repeat_analytes[split])
                else:
                    diff_analytes = repeat_analytes[split].symmetric_difference(
                        repeat_1_analytes[split]
                    )
                    if diff_analytes != set():
                        # Doesn't raise an exception as don't want to prevent
                        # analysis with unbalanced classes, but do want the user
                        # to check that this is what they are expecting!
                        print('\x1b[31m WARNING: The analytes tested in different '
                              'repeats are not the same. Analytes {} are not '
                              'consistent between {} and {} \033[0m'.format(
                              diff_analytes, repeat_1, repeat))
                    orig_analytes += [analyte for analyte in repeat_analytes[split]
                                      if not analyte in orig_analytes]

            # Analytes across the different splits must be identical
            split_analytes = tuple(
                tuple(sub_list) for sub_list in repeat_analytes.values()
            )
            if len(set(split_analytes)) != 1:
                raise Exception(
                    'Analytes across splits in repeat {} are not identical:\n'
                    '{}'.format(repeat, repeat_analytes)
                )

        # Checks that all control analytes (slated for removal) are actually
        # in the dataset
        for analyte in self.control_analytes:
            if not analyte in orig_analytes:
                print('\x1b[31m WARNING: Whilst removing analytes specified'
                      ' by the user to ignore, analyte {} is not '
                      'recognised \033[0m'.format(analyte))
        upd_analytes = sorted([analyte for analyte in orig_analytes
                               if not analyte in self.control_analytes])

        # Removes analytes the user has specified to ignore in the analysis
        for repeat in scaled_data.keys():
            for split in scaled_data[repeat].keys():
                for plate_path in scaled_data[repeat][split].keys():
                    plate = copy.deepcopy(scaled_data[repeat][split][plate_path])
                    for analyte, fluor_df in plate.items():
                        if analyte in self.control_analytes:
                            del scaled_data[repeat][split][plate_path][analyte]

        # Saves (scaled) plate data as object attributes
        self.plates = raw_plate_data
        self.scaled_data = scaled_data
        features = [
            '{}_{}'.format(split, feature) for split, feature_list in
            self.peptide_dict.items() for feature in feature_list
        ]
        drop_features = [
            '{}_{}'.format(split, feature) for split, feature_list in
            self.control_peptides.items() for feature in feature_list
        ]
        drop_features += ['{}_{}'.format(key, value) for key, value in no_pep.items()]
        self.all_features = [feature for feature in features if not
                             feature in drop_features]
        split_features = OrderedDict()
        for split in self.peptide_dict:
            split_features[split] = []
        for feature in self.all_features:
            split = feature.split('_')[0]
            sub_feature = feature.split('_')[-1]
            if sub_feature in self.peptide_dict[split]:
                split_features[split].append(feature)
            else:
                raise ValueError(
                    'Neither split nor peptide names should contain the '
                    'character "_"'
                )
        self.split_features = split_features
        self.analytes = upd_analytes

        # Records outlier values
        with open('{}/Outliers_same_plate.txt'.format(self.results_dir), 'w') as f:
            f.write('Outliers on same plate identified by generalised ESD test:\n')
            print('\n\nOutliers on the same plate:\n')
            for repeat in outliers.keys():
                for split in outliers[repeat].keys():
                    for plate_path in outliers[repeat][split].keys():
                        for outlier in outliers[repeat][split][plate_path]:
                            analyte = outlier[0]
                            peptide = outlier[1]
                            reading = outlier[2]
                            f.write('{}: {}, {}, {}\n'.format(
                                plate_path, analyte, peptide, reading
                            ))
                            print('{}: {}, {}, {}'.format(
                                plate_path, analyte, peptide, reading
                            ))

    def combine_plate_readings(
        self, outlier_excl_thresh=0.05, drop_thresh=2, k_max=np.nan,
        same_num_repeats=True
    ):
        """
        Calculates the median of repeat readings for each independent sample,
        then combines these median readings into a single dataframe in an
        appropriate format for ML with the functions in train.py

        Input
        ----------
        - outlier_excl_thresh: Significance level for discarding outliers (via a
        generalised ESD test), default value is 0.05
        - drop_thresh: The minimum number of outlier readings a sample requires
        to be discarded from the dataset (only applied when remove_outliers is
        set to True), default value is 2
        - k_max: The maximum number of outliers to exclude with a generalised ESD
        test. If leave as default np.nan, an appropriate threshold will be selected
        based upon the size of the dataset.
        - same_num_repeats: Boolean, describes if the same number of technical
        replicate readings have been collected for all analytes in the same repeat
        """

        scaled_merged_dfs = OrderedDict()
        # Group dataframes for the same analyte together. Check whether
        # features are the same or different - if the former, merge
        # dataframes along axis=0, if the latter merge along axis=1.
        for repeat in self.scaled_data.keys():
            scaled_merged_dfs[repeat] = OrderedDict()
            analyte_dfs_dict = OrderedDict()

            for split in self.scaled_data[repeat].keys():
                repeat_analytes = []

                for plate_path, plate_dict in self.scaled_data[repeat][split].items():
                    for analyte in plate_dict.keys():
                        if not analyte in repeat_analytes:
                            repeat_analytes.append(analyte)

                for analyte in self.analytes:  # Ensures analytes are always
                # considered (and recorded) in the same order
                    if analyte in repeat_analytes:
                        if not analyte in analyte_dfs_dict.keys():
                            analyte_dfs_dict[analyte] = OrderedDict()
                        analyte_dfs_dict[analyte][split] = []

                for plate_path, plate_dict in self.scaled_data[repeat][split].items():
                    for analyte, orig_analyte_df in plate_dict.items():
                        analyte_df = copy.deepcopy(orig_analyte_df).drop(
                            ['Analyte', 'Plate'], axis=1
                        )
                        cols = list(analyte_df.columns)
                        match = False
                        if cols == self.split_features[split]:
                            match = True
                            analyte_dfs_dict[analyte][split].append(analyte_df)
                        if match is False:
                            raise Exception(
                                'Peptides\n{}\ndo not match any of the expected '
                                'peptide lists:\n'
                                '{}'.format(cols, self.split_features[split])
                            )

            for analyte in analyte_dfs_dict.keys():
                # Merges same analyte, same barrels
                comb_df_list = []
                for split in analyte_dfs_dict[analyte].keys():
                    if analyte_dfs_dict[analyte][split] == []:
                        raise Exception(
                            'Data not recorded across all splits for repeat '
                            '{}'.format(repeat)
                        )
                    analyte_df_list = copy.deepcopy(analyte_dfs_dict[analyte][split])
                    comb_df = pd.concat(analyte_df_list, axis=0).reset_index(drop=True)
                    if comb_df.shape[1] != analyte_df_list[0].shape[1]:
                        raise Exception(
                            'Columns in dataframes for analyte: {}, split : {},'
                            ' repeat: {} don\'t match - check that the peptides'
                            ' on the different plates '
                            'match'.format(analyte, split, repeat)
                        )
                    comb_df_list.append(comb_df)

                lens = []
                all_cols = []
                for comb_df in comb_df_list:
                    lens.append(comb_df.shape[0])
                    all_cols += list(comb_df.columns)
                if len(set(lens)) != 1:
                    if same_num_repeats is True:
                        raise Exception(
                            'Different number of replicates measured for {} in '
                            'repeat {} across barrels in different splits:\n{} '
                            'measurements found respectively for {}'.format(
                                analyte, repeat, lens, self.peptide_dict.keys()
                            )
                        )
                    else:
                        max_len = max(lens)
                        for comb_df in comb_df_list:
                            if comb_df.shape[0] < max_len:
                                diff = max_len - comb_df.shape[0]
                                for n in range(diff):
                                    comb_df = comb_df.append(pd.Series(
                                        [np.nan for i in range(comb_df.shape[1])],
                                        index=comb_df.columns, name='PlaceHolder'
                                    ))
                                comb_df = comb_df.reset_index(drop=True)
                if len(set(all_cols)) != len(all_cols):
                    raise Exception(
                        'Repeated columns found across splits:\n'
                        '{}'.format(self.peptide_dict)
                    )

                # Merges same analyte, different barrels
                comb_df = pd.concat(comb_df_list, axis=1).reset_index(drop=True)
                scaled_merged_dfs[repeat][analyte] = comb_df

        outliers = OrderedDict()
        median_dfs_list = []
        labels_list = []
        # Calculates median barrel readings for each analyte for each repeat.
        # Will raise an error if any of the median values are calculated to be
        # NaN.
        for repeat in scaled_merged_dfs.keys():
            for analyte, orig_analyte_df in scaled_merged_dfs[repeat].items():
                analyte_df = copy.deepcopy(orig_analyte_df)
                analyte_df['Analyte'] = [analyte for n in range(analyte_df.shape[0])]
                outliers['{}_{}'.format(repeat, analyte)] = []
                outliers['{}_{}'.format(repeat, analyte)] = highlight_outliers(
                    analyte_df, remove_outliers=False,
                    plate_outliers=outliers['{}_{}'.format(repeat, analyte)],
                    alpha=outlier_excl_thresh, drop_thresh=drop_thresh,
                    user_k_max=k_max
                )

                median_data = calc_median(analyte_df, raise_warning=True)
                median_dfs_list.append(median_data)
                labels_list.append(analyte)

        # Records outlier values
        with open('{}/Outliers_across_plates.txt'.format(self.results_dir), 'w') as f:
            f.write('Outliers across merged plates identified by generalised ESD test:\n')
            print('\n\nOutliers across plates:\n')
            for sample_id in list(outliers.keys()):
                for outlier in outliers[sample_id]:
                    analyte = outlier[0]
                    peptide = outlier[1]
                    reading = outlier[2]
                    f.write('{}: {}, {}, {}\n'.format(
                        sample_id, analyte, peptide, reading
                    ))
                    print('{}: {}, {}, {}'.format(
                        sample_id, analyte, peptide, reading
                    ))

        # DataFrame of readings for all measured samples (before outlier removal)
        orig_fluor_df = pd.concat(median_dfs_list, axis=0).reset_index(drop=True)
        orig_labels_df = pd.DataFrame({'Analyte': labels_list})
        orig_fluor_df = pd.concat(
            [orig_fluor_df, orig_labels_df], axis=1
        ).reset_index(drop=True)

        # Removes outlier samples from orig_fluor_df
        excl_outliers_dfs = []
        with open('{}/Sample_outliers.txt'.format(self.results_dir), 'w') as f:
            f.write(
                'Sample outliers identified by generalised ESD test and '
                'excluded from final output dataset (outliers are represented '
                'as NaN):\n'
            )
        print('\n\nSample outliers:\n')

        for analyte in self.analytes:
            analyte_indices = [i for i, val in enumerate(labels_list) if val == analyte]
            analyte_df = copy.deepcopy(
                orig_fluor_df
            ).iloc[analyte_indices].reset_index(drop=True)
            sub_analyte_df, analyte_outliers = highlight_outliers(
                analyte_df, remove_outliers=True, plate_outliers=[],
                alpha=outlier_excl_thresh, drop_thresh=drop_thresh,
                user_k_max=k_max
            )
            # Prints outliers dropped from the final dataset
            for index, count_tup in analyte_outliers.items():
                count = count_tup[0]
                if count >= drop_thresh:
                    drop_indices = count_tup[1]
                    nan_df = copy.deepcopy(analyte_df.iloc[index,:])
                    for nan_index in drop_indices:
                        orig_val = copy.deepcopy(nan_df.iloc[nan_index])
                        nan_df.iloc[nan_index] = 'Outlier: {}'.format(orig_val)
                    print(
                        '\x1b[31m Outlier excluded from final output dataset: {}, {}'
                        '\n{}\n \033[0m'.format(index, analyte, nan_df)
                    )
                    with open('{}/Sample_outliers.txt'.format(self.results_dir), 'a') as f:
                        f.write(
                            '{}, {}: {}\n\n'.format(index, analyte, nan_df)
                        )
            excl_outliers_dfs.append(sub_analyte_df.reset_index(drop=True))
        comb_fluor_df = pd.concat(excl_outliers_dfs, axis=0).reset_index(drop=True)

        self.orig_fluor_data = orig_fluor_df
        self.ml_fluor_data = comb_fluor_df  # Scale the train and test data
        # separately during ML cross-validation loop => don't perform here!

    def display_data_distribution(
        self, ml_fluor_data=None, results_dir=None, scale=True, cushion=0.2,
        class_order=None, prefix='', test=False
    ):
        """
        Generates plots to represent the range of values recorded across repeats
        (and so inform the user whether they might want to try a different
        scaling method, or exclude particular features)

        Input
        ----------
        - ml_fluor_data: DataFrame of fluorescence readings. If None, will be
        set to self.ml_fluor_data.
        - results_dir: Directory where output plots are to be saved. If None,
        will be set to self.results_dir.
        - scale: Boolean, specifies whether to scale the input dataframe of
        fluorescence readings and compare it to the original (unscaled) data
        - cushion: Size of buffer to add to the y-axes, whose range is between
        (smallest value in dataset - cushion, largest value in dataset +
        cushion)
        - class_order: List of analytes. Provide this list if you want the
        graphs to be drawn in a particular order, otherwise this function will
        automatically determine the analytes present (and plot them in the order
        obtained using the sorted() function)
        - prefix: Prefix to add to the file names of saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False
        """

        if ml_fluor_data is None:
            ml_fluor_data = self.ml_fluor_data
        if results_dir is None:
            results_dir = self.results_dir

        # Draws heatmap fingerprint
        draw_heatmap_fingerprints(
            ml_fluor_data, results_dir, scale, class_order, prefix,
            test
        )
        # Draws boxplots
        draw_boxplots(
            ml_fluor_data, results_dir, scale, cushion, class_order,
            prefix, test
        )

    def run_anova(self):
        """
        Runs one-way anova to determine if there is a significant difference
        between a barrel's fluorescence readings in the absence and presence of
        analyte

        Output
        ----------
        stats_dict: Dictionary listing F and p values output from running
        one-way ANOVA to determine whether there is a significant difference
        between the readings measured for the different analytes for a
        particular peptide
        """

        print(
            'Running one-way ANOVA.\nANOVA assumes that, for each individual '
            'peptide:\n1: The readings for each analyte are normally '
            'distributed\n2: The standard deviations of the readings for each '
            'analyte are equal\n3: The readings are independent'
        )

        stats_dict = OrderedDict()
        median_vals_dict = OrderedDict()

        for peptide in self.all_features:
            stats_dict[peptide] = OrderedDict()
            stats_dict[peptide]['F statistic'] = np.nan
            stats_dict[peptide]['p value'] = np.nan
            median_vals = []
            median_vals_dict[peptide] = OrderedDict()

            for analyte in self.analytes:
                sub_df = copy.deepcopy(self.ml_fluor_data)
                sub_df = sub_df[sub_df['Analyte'] == analyte].reset_index(drop=True)
                median_vals.append(sub_df[peptide].tolist())
                median_vals_dict[peptide][analyte] = sub_df[peptide].tolist()

            (
                stats_dict[peptide]['F statistic'], stats_dict[peptide]['p value']
            ) = f_oneway(*median_vals)

        print('\n\nFor each barrel, is there a significant difference between '
              'the median reading for each analyte?')
        for peptide in stats_dict.keys():
            print('\n\n\n{}'.format(peptide))
            for analyte in median_vals_dict[peptide].keys():
                print('{}: Standard deviation {}'.format(
                    analyte, np.std(median_vals_dict[peptide][analyte], ddof=1)
                ))
            print('{} p-value: {}'.format(peptide, stats_dict[peptide]['p value']))

        return stats_dict
