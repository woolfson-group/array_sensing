

# Functions to parse data collected from plate reader and manipulate it into a
# suitable format to be fed into sklearn ML algorithms.

import copy
import os
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

def parse_xlsx_to_dataframe(plate_path, peptide_list, gain):
    """
    Converts an excel xlsx file output from the plate reader into a pandas
    dataframe.

    The layout of analytes and peptides on the plate is read from the "Plate
    Information" sheet in the xlsx file, and used to group rxc blocks of
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
        plate_path, sheet_name='Protocol Information', index_col=0
    )
    start_row = protocol_df.index.tolist().index('Plate layout')
    label_df = pd.read_excel(
        plate_path, sheet_name='Protocol Information', skiprows=start_row+2,
        nrows=2, index_col=0, usecols=range(13)
    )
    label_df = label_df.replace(
        to_replace='[bB][lL][aA][nN][kK]', value='blank', regex=True
    )

    label_array = label_df.to_numpy()
    label_set = list(set([col for row in label_array for col in row]))

    # Reads peptide layout and names from metadata in "Protocol Information"
    # sheet
    start_row = protocol_df.index.tolist().index('Peptide layout')
    peptide_arrang = pd.read_excel(
        plate_path, sheet_name='Protocol Information', skiprows=start_row+1,
        nrows=8,  index_col=None, usecols=range(2)
    )
    r = peptide_arrang.shape[0]
    c = peptide_arrang.shape[1]
    peptide_arrang = peptide_arrang.to_numpy().flatten('F')

    # Checks that peptides listed in the xlsx file are the same as those input
    # by the user
    if sorted(peptide_list) != sorted(peptide_arrang):
        raise PlateLayoutError('Peptides tested in {} don\'t match peptides '
                               'specified by user'.format(plate_path))

    # Combines fluorescence data collected for the same analyte into a
    # dataframe
    grouped_fluor_data = {}

    for analyte in label_set:
        grouped_fluor_data[analyte] = []

    for index, analyte in np.ndenumerate(label_array):
        index_r = index[0]
        index_c = index[1]

        rows = np.arange((r*index_r), ((r*index_r)+(r+1)))  # + (r+1) so that
        # row range covers r rows
        columns = np.arange(c*(index_c), ((c*index_c)+(c+1)))  # + (c+1) so
        # that column range covers 2 columns

        sub_fluor_df = fluor_df.iloc[rows[0]:rows[-1], columns[0]:columns[-1]]
        sub_fluor_vals = sub_fluor_df.to_numpy().flatten('F')

        # Ensures the order of the peptide labels provided by the user matches
        # the order listed in the input xlsx file
        ordered_fluor_vals = []
        for peptide in peptide_list:
            position = np.asarray(peptide_arrang == peptide).nonzero()[0]
            if len(position) > 1:
                raise PlateLayoutError(
                    'Barrel {} listed more than once in peptide arrangement '
                    'for plate {}'.format(peptide, plate_path)
                )
            elif len(position) < 1:
                raise PlateLayoutError(
                    'Barrel {} not listed in peptide arrangement for plate '
                    '{}'.format(peptide, plate_path)
                )
            ordered_fluor_vals.append(sub_fluor_vals[position[0]])

        count = len(grouped_fluor_data[analyte])+1
        grouped_fluor_data[analyte].append(pd.DataFrame({count: ordered_fluor_vals}))

    for analyte in label_set:
        analyte_df = pd.concat(
            grouped_fluor_data[analyte], axis=1, ignore_index=True
        ).transpose().reset_index(drop=True)
        analyte_df.columns = peptide_list
        grouped_fluor_data[analyte] = analyte_df

    return grouped_fluor_data


def check_for_saturation(saturated_readings, plate, plate_path, min_fluor,
                         max_fluor):
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
    - min_fluor: Minimum fluoresence reading the plate can measure
    - max_fluor: Maximum fluorescence reading the plate can measure

    Output
    ----------
    - saturated_readings: Updated list of saturated measurements, identified by
    plate file path and barrel name
    """

    for analyte, fluor_df in plate.items():
        fluor_array = fluor_df.to_numpy()

        for index, val in np.ndenumerate(fluor_array):
            if val <= min_fluor or val >= max_fluor:
                data_point = '{}_{}'.format(plate_path, fluor_df.columns[index[1]])
                if not data_point in saturated_readings:
                    saturated_readings.append(data_point)

    return saturated_readings


def remove_outliers_and_average(input_df):
    """
    Calculates the average of each column in a dataframe (ignoring data points
    outside of 95% confidence limits (2-tailed)).

    Input
    ----------
    - input_df: DataFrame, dimensions = n_samples x n_features

    Output
    ----------
    - output_ser: Series, dimensions = n_features
    """

    output_df = input_df[
        np.abs(input_df - input_df.mean()) <= (1.96*input_df.std())
    ]
    output_ser = output_df.mean()

    return output_ser


def scale_min_max(plate, no_pep):
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
    blank_data = remove_outliers_and_average(blank_data)

    scaled_plate = {}
    analytes = [analyte for analyte in list(plate.keys()) if analyte != 'blank']
    for analyte in analytes:
        fluor_data = copy.deepcopy(plate[analyte])

        # Calculates min fluorescence reading for the analyte
        min_fluor = float(remove_outliers_and_average(fluor_data[[no_pep]]))
        min_avrg = float(fluor_data[[no_pep]].mean())
        min_std = float(fluor_data[[no_pep]].std())

        del fluor_data[no_pep]

        for index_c, peptide in enumerate(list(fluor_data.columns)):
            if peptide.strip().upper() == 'GRP35':
                grp35 = peptide  # Used to delete collapsed barrel readings
                """
                # Checks that GRP35 data (collapsed barrel) is within 95%
                # confidence limits of no peptide data
                for val in fluor_data[peptide].tolist():
                    if np.abs(val-min_avrg) > (1.96*min_std):
                        print('Unusual GRP35 (collapsed barrel) reading {} '
                              'for analyte {}'.format(val, analyte))
                        usr_input = input('Continue with analysis?\n').lower()
                        while not usr_input in ['yes', 'y', 'no', 'n']:
                            usr_input = input('Input not recognised - please '
                                              'enter "yes" or "no":\n')
                """

            else:
                # Applies min max scaling to each reading
                max_fluor = blank_data[peptide]

                for index_r, val in enumerate(fluor_data[peptide].tolist()):
                    scaled_val = (val-min_fluor) / (max_fluor-min_fluor)
                    fluor_data.iloc[index_r, index_c] = scaled_val

        del fluor_data[grp35]  # Removes collapsed barrel (experimental control only)
        scaled_plate[analyte] = fluor_data

    return scaled_plate, grp35


class def_data():

    def __init__(self, dir_path, repeat_names, peptide_list):
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
            raise PlateLayoutError(
                'Barrel {} listed more than once'.format(peptide)
            )


class parse_array_data(def_data):


    def __init__(self, dir_path, repeat_names, peptide_list, gain=1,
                 min_fluor=0, max_fluor=260000):
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
        def_data.__init__(self, dir_path, repeat_names, peptide_list)
        self.gain = gain
        self.min_fluor = min_fluor
        self.max_fluor = max_fluor

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
                        file_name.endswith('xlsx')
                    and not file_name.startswith('~$')
                    and repeat in file_name
                ):
                    ind_meas_xlsx[repeat].append(file_name)

        self.ind_meas_xlsx = ind_meas_xlsx

    def xlsx_to_scaled_df(self, no_pep):
        """
        Wrapper function to parse grouped xlsx files into dataframes and
        perform min max scaling.

        Input
        --------
        - no_pep: Name of sample without peptide in peptide layout
        """

        plates = {}
        ind_meas_df = {}
        saturated_readings = []

        for repeat, xlsx_list in self.ind_meas_xlsx.items():
            plates[repeat] = []
            ind_meas_df[repeat] = []  # List rather than dictionary because
            # analyte name(s) is/are read from the plate not the file name

            for xlsx in xlsx_list:
                plate_path = self.dir_path + xlsx
                plate = parse_xlsx_to_dataframe(plate_path, self.peptides, self.gain)
                plates[repeat].append(plate)
                saturated_readings = check_for_saturation(
                    saturated_readings, plate, plate_path, self.min_fluor,
                    self.max_fluor
                )
                scaled_plate, grp35 = scale_min_max(plate, no_pep)
                ind_meas_df[repeat].append(scaled_plate)

        if saturated_readings != []:
            saturated_readings = '\n'.join(saturated_readings)
            raise FluorescenceSaturationError(
                'The following data points appear to have saturated the plate '
                'reader:\n{}\nTo proceed with analysis you need to either remove'
                ' the identified barrels, or replace the fluorescence\nreadings'
                ' for this barrel with readings collected at a different gain, '
                'for ALL xlsx files in the dataset.'.format(saturated_readings))

        self.plates = plates
        self.ind_meas_df = ind_meas_df
        self.features = [feature for feature in self.peptides
                         if not feature in [no_pep, grp35]]

    def combine_plate_readings(self):
        """
        Combines independent measurements into a single array
        """

        fluor_data = []
        labels = []

        for repeat, plate_list in self.ind_meas_df.items():
            analytes = set([analyte for plate in plate_list for analyte in
                            list(plate.keys())])

            for analyte in analytes:
                # Groups dataframes measuring the same analyte together
                analyte_dfs = []

                for plate in plate_list:
                    if analyte in list(plate.keys()):
                        analyte_dfs.append(copy.deepcopy(plate[analyte]))

                analyte_df = pd.concat(analyte_dfs, axis=0).reset_index(drop=True)
                readings = remove_outliers_and_average(analyte_df).tolist()

                fluor_data.append(readings)
                labels.append(analyte)

        fluor_df = pd.DataFrame(fluor_data)
        labels_df = pd.DataFrame({'Analyte': labels})
        fluor_df = pd.concat([fluor_df, labels_df], axis=1)
        fluor_df.columns = self.features + ['Analyte']
        self.fluor_data = fluor_df

    def standardise_readings(self):
        """
        Standardises fluorescence data across features (i.e. the peptides)
        """

        analyte_labels = self.fluor_data.loc[:, 'Analyte']
        fluor_data = self.fluor_data.drop('Analyte', axis=1)
        standardised_fluor_data = preprocessing.scale(
            fluor_data.to_numpy(), axis=0
        )  # Normalises (by subtracting the mean and dividing by the standard
        # deviation) the readings for each peptide, as required by several ML
        # algorithms
        standardised_fluor_df = pd.concat(
            [pd.DataFrame(fluor_data), analyte_labels], axis=1
        )
        standardised_fluor_df.columns = self.features + ['Analyte']
        self.standardised_fluor_data = standardised_fluor_df
