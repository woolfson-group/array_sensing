

# Functions to perform ML analysis on parsed BADASS data.

import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from matplotlib.colors import BASE_COLORS, CSS4_COLORS
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_selection import (
    f_classif, mutual_info_classif, SelectKBest
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler

sns.set()

if __name__ == 'subroutines.train':
    from subroutines.parse_array_data import DefData
    from subroutines.exceptions import AlgorithmError
else:
    from array_sensing.subroutines.parse_array_data import DefData
    from array_sensing.subroutines.exceptions import AlgorithmError


def bootstrap_data(x, y, features, scale):
    """
    Generates bootstrapped repeat of input arrays (x and y)

    Input
    ----------
    - x: Numpy array of x values. By default set to self.x.
    - y: Numpy array of y values. By default set to self.y.
    - features: List of input features corresponding to the columns in x. By
    default set to self.features.
    - scale: Boolean, defines whether to scale the data (by subtracting the
    median and dividing by the IQR) before calculating feature importances.
    By default is set to True since scaling the data will affect the
    importance scores.

    Output
    ----------
    temp_x: Bootstrapped x array
    temp_y: Bootstrapped y array
    """

    random_rows = []
    temp_x = pd.DataFrame(
        data=copy.deepcopy(x), index=None, columns=features
    )
    temp_y = pd.DataFrame(
        data=copy.deepcopy(y), index=None, columns=['Analyte']
    )
    for m in range(temp_x.shape[0]):
        random_rows.append(random.randint(0, (temp_x.shape[0]-1)))
    temp_x = temp_x.iloc[random_rows,:].reset_index(drop=True)
    temp_y = temp_y.iloc[random_rows,:].reset_index(drop=True)['Analyte'].tolist()

    if scale is True:
        temp_x_scaled = RobustScaler().fit_transform(temp_x)
        temp_x = pd.DataFrame(temp_x_scaled, index=None, columns=features)

    return temp_x, temp_y


def make_feat_importance_plots(
    feature_importances, results_dir, plt_name, test
):
    """
    Generates plots for feature importance score functions

    Input
    ----------
    - feature_importances: Dictionary of importance scores for the different
    features included in the dataset
    - results_dir: Directory where the plots are to be saved
    - plt_name: Prefix to append to the names of the saved plots
    - test: Boolean describing whether the function is being run during the
    program's unit tests - by default is set to False

    Output
    ----------
    - importance_df: DataFrame of features and their importance scores
    """

    cols = []
    cols_all = []
    all_vals = []
    median_vals = []
    lower_conf_limit_vals = []
    upper_conf_limit_vals = []
    for col, importances in feature_importances.items():
        cols.append(col)
        median_vals.append(np.median(importances))
        lower_conf_limit_vals.append(np.percentile(importances, 2.5))
        upper_conf_limit_vals.append(np.percentile(importances, 97.5))
        for importance in importances:
            cols_all.append(col)
            all_vals.append(importance)

    plt.clf()
    plt.figure(figsize=(15,6))
    sns.barplot(x=cols, y=median_vals)
    sns.stripplot(x=cols, y=lower_conf_limit_vals, edgecolor='k',
                  linewidth=1, s=6, jitter=False)
    sns.stripplot(x=cols, y=upper_conf_limit_vals, edgecolor='k',
                  linewidth=1, s=6, jitter=False)
    plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
    plt.xlabel('Peptide')
    plt.ylabel('Importance')
    plt.savefig('{}/{}_feat_importance_percentiles.svg'.format(
        results_dir, plt_name
    ))
    if test is False:
        plt.show()

    plt.clf()
    plt.figure(figsize=(15,6))
    sns.stripplot(x=cols_all, y=all_vals, edgecolor='k', linewidth=1,
                  size=2.5, alpha=0.2)
    plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
    plt.xlabel('Peptide')
    plt.ylabel('Importance')
    plt.savefig('{}/{}_feat_importance_all_data.svg'.format(
        results_dir, plt_name
    ))
    if test is False:
        plt.show()

    importance_df = pd.DataFrame({'Feature': cols, 'Score': median_vals})
    importance_df = importance_df.sort_values(
        by=['Score'], axis=0, ascending=False
    ).reset_index(drop=True)

    return importance_df


class ManualFeatureSelection(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, all_features, selected_features):
        """
        """

        self.all_features = all_features
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        """

        df = pd.DataFrame(data=X, index=None, columns=self.all_features)
        sub_df = df[self.selected_features]
        X = sub_df.to_numpy()

        return X


class RunML(DefData):

    def __init__(
        self, results_dir, fluor_data, classes=None, subclasses=None,
        shuffle=True
    ):
        """
        - results_dir: Path (either absolute or relative) to directory where
        output files should be saved. This directory will be created by the
        program and so should not already exist.
        - fluor_data: DataFrame of (scaled, but not standardised) fluorescence
        readings (output from parse_array_data class)
        - classes: List defining class labels associated with the samples, e.g.
        for analysis of tea types an example list would be ['Green', 'Green',
        'Black', 'Black', 'Grey', 'Grey']. If not defined, code will select the
        column labelled 'Analyte' in the fluor_data DataFrame. Default set to
        None (i.e. not defined).
        - subclasses: List defining subclass labels associated with the samples
        (if such subclass labels exist), e.g. for analysis of tea types an
        example list would be ['Green_Tetleys', 'Green_Pukka',
        'Black_Yorkshire', 'Black_Yorkshire', 'Grey_Asda', 'Grey_Pukka'].
        Default set to None (i.e. not defined).
        - shuffle: Boolean, setting to True directs the program to randomly
        shuffle the input data before ML (to ensure removal of any ordering
        effects in the input data.) Default set to True.
        """

        DefData.__init__(self, results_dir)

        # Checks that if subclasses are defined, classes are also defined
        if not subclasses is None and classes is None:
            raise TypeError(
                '"subclasses" argument cannot be set to a value other than None'
                ' if the "classes" argument is set equal to None'
            )
        # Checks that classes argument is the correct type
        if not classes is None:
            if not type(classes) is list:
                raise TypeError(
                    'Expect "classes" argument to be set either to None or to a'
                    ' list, instead is {}'.format(type(classes))
                )
        # Checks that subclasses argument is the correct type
        if not subclasses is None:
            if not type(subclasses) is list:
                raise TypeError(
                    'Expect "subclasses" argument to be set either to None or '
                    'to a list, instead is {}'.format(type(subclasses))
                )
            for subclass in subclasses:
                if subclass.count('_') != 0:
                    raise ValueError(
                        'Entries in subclass list should be formatted as '
                        '"class_subclass" (in which neither "class" nor '
                        '"subclass" contains the character "_")'
                    )

        # Defines classes and subclasses lists if not specified by the user
        if classes is None:
            try:
                classes = fluor_df['Analyte'].tolist()
            except KeyError(
                'No "Analyte" column detected in input dataframe - if you do '
                'not define the "classes" argument, the input dataframe must '
                'contain an "Analyte" column'
            )
        if subclasses is None:
            subclasses = [np.nan for n in range(fluor_df.shape[0])]
        if len(classes) != fluor_df.shape[0]:
            raise ValueError(
                'Mismatch between number of entries in the input dataframe and '
                'the "classes" list'
            )
        if len(subclasses) != fluor_df.shape[0]:
            raise ValueError(
                'Mismatch between number of entries in the input dataframe and '
                'the "subclasses" list'
            )

        # Ensures that no columns in the input dataframe will be overwritten
        # when 'Classes' and 'Subclasses' columns are subsequently appended
        if any(x in fluor_data.columns for x in ['Classes', 'Subclasses']):
            raise NameError(
                'Please rename any columns in input dataframe labelled either '
                '"Classes" or "Subclasses", as these columns are added to the '
                'dataframe by the code during data processing'
            )
        class_df = pd.DataFrame({'Classes': classes,
                                 'Subclasses': subclasses})
        fluor_data = fluor_data.reset_index(drop=True)
        fluor_data = pd.concat(
            [fluor_data, class_df], axis=1
        ).reset_index(drop=True)

        # Randomly shuffles input data
        if shuffle is True:
            indices = [i for i in range(fluor_data.shape[0])]
            random.shuffle(indices)
            fluor_data = fluor_data.iloc[indices].reset_index(drop=True)

        # Defines object attributes used in ML functions
        self.classes = fluor_data['Classes'].to_numpy(dtype=str)
        self.sub_classes = fluor_data['Subclasses'].to_numpy(dtype=str)
        fluor_data = fluor_data.drop(['Classes', 'Subclasses'], axis=1)
        try:
            fluor_data = fluor_data.drop(['Analyte'], axis=1)
        except KeyError:
            pass
        self.fluor_data = fluor_data
        self.features = self.fluor_data.columns.tolist()
        self.x = self.fluor_data.to_numpy()
        self.y = copy.deepcopy(self.classes)

        # Checks there are no NaN values in the input data
        if self.fluor_data.isna().any(axis=None):
            raise ValueError(
                'NaN detected in input dataframe'
            )
        if np.dtype('O') in self.fluor_data.dtypes.tolist():
            raise ValueError(
                'Non-numeric value detected in input dataframe'
            )
        if 'nan' in self.classes:
            raise ValueError(
                'NaN detected in class values'
            )

        if 'nan' in self.sub_classes:
            nan_array = np.array(['nan' for n in range(self.fluor_data.shape[0])])
            if (self.sub_classes == nan_array).all(axis=None):
                self.sub_classes = None
            else:
                raise ValueError(
                    'NaN detected in subclass values'
                )

    def split_train_test_data_random(self, percent_test=0.2, test=False):
        """
        Splits data at random into training and test sets

        Input
        ----------
        - percent_test: Float in the range 0-0.5, defines the fraction of the
        total data to be set aside for model testing
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False
        """

        if not type(percent_test) in [int, float]:
            raise TypeError(
                '"percent_test" argument should be set to a float in the range '
                '0 - 0.5'
            )
        else:
            if percent_test < 0 or percent_test > 0.5:
                raise ValueError(
                    '"percent_test" argument should be set to a float in the '
                    'range 0 - 0.5'
                )

        if percent_test > 0:
            if test is False:
                skf = StratifiedKFold(
                    n_splits=round(1/percent_test), shuffle=True
                )
            else:
                skf = StratifiedKFold(
                    n_splits=round(1/percent_test), shuffle=True, random_state=0
                )
            split = list(skf.split(X=self.x, y=self.y))[0]
        elif percent_test == 0:
            split = [np.array([n for n in range(self.x.shape[0])]), np.array([])]

        train_indices = split[0]
        test_indices = split[1]

        # Defines train and test datasets
        self.train_data = copy.deepcopy(
            self.fluor_data
        ).iloc[train_indices].reset_index(drop=True)
        self.test_data = copy.deepcopy(
            self.fluor_data
        ).iloc[test_indices].reset_index(drop=True)
        self.train_x = copy.deepcopy(self.x)[train_indices]
        self.test_x = copy.deepcopy(self.x)[test_indices]
        self.train_y = copy.deepcopy(self.y)[train_indices]
        self.test_y = copy.deepcopy(self.y)[test_indices]
        if self.sub_classes is None:
            self.train_groups = None
            self.test_groups = None
        else:
            self.train_groups = copy.deepcopy(self.sub_classes)[train_indices]
            self.test_groups = copy.deepcopy(self.sub_classes)[test_indices]

    def split_train_test_data_user_defined(self, test_subclasses):
        """
        Splits data into training and test sets defined by the user. Will only
        run if self.sub_classes is not set to None.

        Input
        ----------
        - test_subclasses: List of the classes to be separated out for the test
        set. Must not include all of the subclasses in a class
        """

        if self.sub_classes is None:
            raise ValueError(
                'Function separates different subclasses into training and test'
                ' sets, but no subclasses have been defined - please set a '
                'value for self.sub_classes'
            )

        if not type(test_subclasses) is list:
            raise ValueError(
                'Expect "test_subclasses" argument to be a list, instead have '
                'been provided with {}'.format(type(test_subclasses))
            )

        train_indices = []
        test_indices = []

        for n in range(self.sub_classes.shape[0]):
            if self.sub_classes[n] in test_subclasses:
                test_indices.append(n)
            else:
                train_indices.append(n)

        # Defines train and test datasets
        self.train_data = copy.deepcopy(
            self.fluor_data
        ).iloc[train_indices].reset_index(drop=True)
        self.test_data = copy.deepcopy(
            self.fluor_data
        ).iloc[test_indices].reset_index(drop=True)
        self.train_x = copy.deepcopy(self.x)[train_indices]
        self.test_x = copy.deepcopy(self.x)[test_indices]
        self.train_y = copy.deepcopy(self.y)[train_indices]
        self.test_y = copy.deepcopy(self.y)[test_indices]
        if self.sub_classes is None:
            self.train_groups = None
            self.test_groups = None
        else:
            self.train_groups = copy.deepcopy(self.sub_classes)[train_indices]
            self.test_groups = copy.deepcopy(self.sub_classes)[test_indices]

    def calc_feature_correlations(
        self, train_data=None, correlation_coeff='kendall',
        plt_name='', abs_vals=True, test=False
    ):
        """
        Calculates pairwise Kendall tau rank correlation coefficient values
        between all 2-way combinations of features, and plots a heatmap.

        Input
        ----------
        - train_data: DataFrame of the training data. By default set to
        self.fluor_data
        - correlation_coeff: String defining the coefficient coefficient to be
        calculate between the data, by default is set to Kendall's tau. Can
        alternatively be set to "pearson" (not recommended) or "spearman".
        - plt_name: Prefix to append to name of the saved plot
        - abs_vals: Boolean, if set to True colours the the heatmap by absolute
        correlation coefficient values, if set to False colours by the raw
        values. (Colouring by the absolute values prevents strong negative
        correlations from appearing to be weakly correlated when quickly
        glancing at the heatmap.)
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - correlation matrix: DataFrame of Kendall tau rank correlation
        coefficient values for all pairwise combinations of features
        """

        # Checks arguments are suitable for running the function
        if train_data is None:
            train_data = copy.deepcopy(self.fluor_data)

        if not correlation_coeff in ['kendall', 'spearman', 'pearson']:
            raise ValueError(
                'Value specified for "correlation_coeff" not recognised - '
                'should be set to "kendall", "spearman" or "pearson"'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        if type(abs_vals) != bool:
            raise TypeError(
                '"abs_vals" should be a Boolean value'
            )

        # Calculates the correlation coefficient
        correlation_matrix = copy.deepcopy(train_data).corr(method=correlation_coeff)
        if abs_vals is True:
            correlation_matrix = correlation_matrix.abs()

        plt.clf()
        heatmap = sns.heatmap(
            data=correlation_matrix, cmap='RdBu_r', annot=True,
            xticklabels=True, yticklabels=True, fmt='.3f'
        )
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        plt.savefig('{}/{}Feature_correlations_Kendall_tau_rank.svg'.format(
            self.results_dir, plt_name
        ))
        if test is False:
            plt.show()

        return correlation_matrix

    def calc_feature_importances_kbest(
        self, x=None, y=None, features=None, method_classif='f_classif',
        num_repeats=1000, scale=True, plt_name='', test=False
    ):
        """
        Runs a univariate statistical test (either f_classif (ANOVA F-statistic,
        parametric) or mutual_info_classif (mutual information, non-parametric))
        between x and y to calculate the importances of the different input
        features, and plots these values. Test is run multiple (default 1000)
        times on bootstrapped data, results are then displayed as the median and
        95% confidence limits calculated from the 2.5th and 97.5th percentiles.
        N.B. Assumes that the features are independent of one another.

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of y values. By default set to self.y.
        - features: List of input features corresponding to the columns in x. By
        default set to self.features.
        - method_classif: Either 'f_classif' or 'mutual_info_classif'. From the
        scikit-learn docs - 'The methods based on F-test (f_classif) estimate
        the degree of linear dependency between two random variables. On the
        other hand, mutual information methods can capture any kind of
        statistical dependency, but being nonparametric, they require more
        samples for accurate estimation.' By default set to f_classif
        - num_repeats: The number of times to repeat the test. By default set
        to 1000.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the
        importance scores.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - importance_df: DataFrame of features and their importance scores
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if features is None:
            features = copy.deepcopy(self.features)

        if not method_classif in [
            f_classif, mutual_info_classif, 'f_classif', 'mutual_info_classif'
        ]:
            raise ValueError(
                '"method_classif" should be set to either "f_classif" or '
                '"mutual_info_classif"'
            )
        if method_classif == 'f_classif':
            method_classif = f_classif
        elif method_classif == 'mutual_info_classif':
            method_classif = mutual_info_classif

        if type(num_repeats) != int:
            raise TypeError(
                '"num_repeats" should be set to a positive integer value'
            )
        else:
            if num_repeats <= 0:
                raise ValueError(
                    '"num_repeats" should be set to a positive integer value'
                )

        if type(scale) != bool:
            raise TypeError(
                '"scale" should be set to a Boolean value'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        # Runs SelectKBest
        univ_feature_importances = OrderedDict()
        for col in features:
            univ_feature_importances[col] = [np.nan for n in range(num_repeats)]

        for n in range(num_repeats):
            # Uses bootstrapping to create a "new" dataset
            temp_x, temp_y = bootstrap_data(x, y, features, scale)

            model = SelectKBest(score_func=method_classif, k='all')
            model.fit(X=temp_x, y=temp_y)
            total_y = np.sum(model.scores_)
            norm_y = model.scores_ / total_y

            for col, importance in enumerate(norm_y):
                col = features[col]
                univ_feature_importances[col][n] = importance

        plt_name = '{}_KBest'.format(plt_name)
        importance_df = make_feat_importance_plots(
            univ_feature_importances, self.results_dir, plt_name, test
        )

        return importance_df

    def calc_feature_importances_tree(
        self, x=None, y=None, features=None, num_repeats=1000, scale=True,
        plt_name='', test=False
    ):
        """
        Fits an ExtraTrees classifier to the data (a number of randomised
        decision trees are fitted to various different sub-samples of the data,
        these "sub-trees" are then averaged). Feature importance scores can be
        calculated from the average contribution which each feature makes to the
        predictive accuracy of the sub-trees in which it is included. Classifier
        is fitted multiple (default 1000) times on bootstrapped data, results
        are then displayed as the median and 95% confidence limits calculated
        from the 2.5th and 97.5th percentiles.
        N.B. Does not assume that the features are independent of one another,
        but equally does not fully take into account correlations between
        features (the user should run permutation analysis instead if feature
        correlation is a real concern).

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of y values. By default set to self.y.
        - features: List of input features corresponding to the columns in x. By
        default set to self.features.
        - num_repeats: The number of times to repeat the test. By default set
        to 1000.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the
        importance scores.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - importance_df: DataFrame of features and their importance scores
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if features is None:
            features = copy.deepcopy(self.features)

        if type(num_repeats) != int:
            raise TypeError(
                '"num_repeats" should be set to a positive integer value'
            )
        else:
            if num_repeats <= 0:
                raise ValueError(
                    '"num_repeats" should be set to a positive integer value'
                )

        if type(scale) != bool:
            raise TypeError(
                '"scale" should be set to a Boolean value'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        # Fits ExtraTrees classifiers
        tree_feature_importances = OrderedDict()
        for col in features:
            tree_feature_importances[col] = [np.nan for n in range(num_repeats)]

        for n in range(num_repeats):
            # Uses bootstrapping to create a "new" dataset
            temp_x, temp_y = bootstrap_data(x, y, features, scale)

            model = ExtraTreesClassifier()
            model.fit(X=temp_x, y=temp_y)
            feature_importances = model.feature_importances_

            for col, importance in enumerate(model.feature_importances_):
                col = features[col]
                tree_feature_importances[col][n] = importance

        plt_name = '{}_Tree'.format(plt_name)
        importance_df = make_feat_importance_plots(
            tree_feature_importances, self.results_dir, plt_name, test
        )

        return importance_df

    def calc_feature_importances_permutation(
        self, x=None, y=None, features=None, classifier=AdaBoostClassifier,
        parameters={'n_estimators': [10, 30, 100, 300, 1000]},
        model_metric='accuracy', num_repeats=1000, scale=True, plt_name='',
        test=False
    ):
        """
        Runs permutation analysis with a classifier to calculate feature
        importances (for each feature in turn, the performance of a classifier
        trained with and without that feature on the input dataset is compared -
        the importance score is the difference in performance between the two
        models). Analysis is run multiple (default 1000) times on bootstrapped
        data, results are then displayed as the median and 95% confidence limits
        calculated from the 2.5th and 97.5th percentiles.
        N.B. Does not assume that the features are independent of one another,
        but equally does not fully take into account correlations between
        features (the user should run permutation analysis instead if feature
        correlation is a real concern).

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of y values. By default set to self.y.
        - features: List of input features corresponding to the columns in x. By
        default set to self.features.
        - classifier: The type of classifier to train. By default set to
        AdaBoostClassifier. BEWARE of selecting a classifier that is slow to
        train, especially if num_repeats is set to a high value.
        - parameters: Dictionary of parameter values to optimise (via grid
        search) for the classifier. Also include in this dictionary any fixed,
        non-default parameter values that you want to use. By default set to a
        sensible range of n_estimators to try for an AdaBoostClassifier. BEWARE
        of setting to an extensive range of parameters to test, as the function
        will run *very* slowly.
        - model_metric: The metric to use to measure classifier performance. By
        default set to accuracy.
        - num_repeats: The number of times to repeat the test. By default set
        to 1000.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the
        importance scores.
        - plt_name: Prefix to append to the names of the saved plots
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - importance_df: DataFrame of features and their importance scores
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if features is None:
            features = copy.deepcopy(self.features)

        if type(num_repeats) != int:
            raise TypeError(
                '"num_repeats" should be set to a positive integer value'
            )
        else:
            if num_repeats <= 0:
                raise ValueError(
                    '"num_repeats" should be set to a positive integer value'
                )

        if type(scale) != bool:
            raise TypeError(
                '"scale" should be set to a Boolean value'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        # Fits classifiers
        permutation_feature_importances = OrderedDict()
        for col in features:
            permutation_feature_importances[col] = [np.nan for n in range(num_repeats)]

        # For speed reasons, perform one grid search to obtain "optimal"
        # parameters on the original data, rather than re-running for each
        # bootstrapped dataset => greatly increases function speed whilst having
        # little effect upon performance (an OK set of parameter values is
        # expected to work well for all of the bootstrapped datasets)
        orig_model = classifier()
        orig_grid_search = GridSearchCV(
         estimator=orig_model, param_grid=parameters, error_score=np.nan,
         scoring=model_metric
        )
        orig_grid_search.fit(X=x, y=y)
        best_params = orig_grid_search.best_params_

        for n in range(num_repeats):
            # Uses bootstrapping to create a "new" dataset
            temp_x, temp_y = bootstrap_data(x, y, features, scale)

            model = classifier(**best_params)
            model_boost.fit(temp_x, temp_y)
            results = permutation_importance(
                model, temp_x, temp_y, scoring='accuracy', n_jobs=-1
            )

            for col, importance in enumerate(results.importances_mean):
                col = features[col]
                permutation_feature_importances[col][n] = importance

        plt_name = '{}_Permutation'.format(plt_name)
        importance_df = make_feat_importance_plots(
            permutation_feature_importances, self.results_dir, plt_name, test
        )

        return importance_df

    def run_pca(self, x=None, scale=True, plt_name='', test=False):
        """
        Runs Principal Component Analysis and makes scatter plot of number of
        components vs. amount of information captured

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of y values. By default set to self.y.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the PCA.
        - plt_name: Prefix to append to the names of the saved plot
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - model: PCA model fitted to x values
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if features is None:
            features = copy.deepcopy(self.features)

        if type(scale) != bool:
            raise TypeError(
                '"scale" should be set to a Boolean value'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        # Runs PCA
        if scale is True:
            x = RobustScaler().fit_transform(x)

        model = PCA()
        model.fit(x)
        max_num_components = len(model.explained_variance_ratio_)

        plt.clf()
        x_labels = np.linspace(1, max_num_components, max_num_components)
        lineplot = sns.lineplot(
            x_labels, np.cumsum(model.explained_variance_ratio_), marker='o'
        )
        plt.xticks(
            np.arange(1, x_labels.shape[0]+1), x_labels, rotation='vertical'
        )
        plt.yticks(rotation='horizontal')
        plt.savefig('{}/{}PCA_scree_plot.svg'.format(self.results_dir, plt_name))
        if test is False:
            plt.show()

        return model

    def plot_scatter_on_pca_axes(
        self, x=None, y=None, subclasses=None, num_dimensions=2, scale=True,
        plt_name='', test=False
    ):
        """
        Generates scatter plot of input data on axes corresponding to the first
        two (2D plot)/three (3D) principal components calculated for the dataset

        Input
        ----------
        - x: Numpy array of x values. By default set to self.x.
        - y: Numpy array of class (y) values. By default set to self.y.
        - subclasses: Numpy array of subclasses. By default set to
        self.sub_classes.
        - num_dimensions: The number of PCA dimensions to incorporate in the
        scatter plot. Must be set to either 2 or 3, by default is set to 2.
        - scale: Boolean, defines whether to scale the data (by subtracting the
        median and dividing by the IQR) before calculating feature importances.
        By default is set to True since scaling the data will affect the PCA.
        - plt_name: Prefix to append to the names of the saved plot
        """

        # Checks argument values are suitable for running the function
        if x is None:
            x = copy.deepcopy(self.x)
        if y is None:
            y = copy.deepcopy(self.y)
        if subclasses is None:
            subclasses = copy.deepcopy(self.sub_classes)
        if not subclasses is None:
            if type(subclasses) != np.ndarray:
                raise TypeError(
                    'Expect "sub_classes" to be set either to None, or to a '
                    '(1D) array of the subclasses present in the dataset'
                )

        if not num_dimensions in [2, 3]:
            raise ValueError(
                'Expect "num_dimensions" to be set to 2 or 3'
            )

        if type(scale) != bool:
            raise TypeError(
                '"scale" should be a Boolean (True or False)'
            )

        if type(plt_name) != str:
            raise TypeError(
                '"plt_name" should be a string value'
            )

        # Selects marker colour and symbol
        colours = [key for key, val in BASE_COLORS.items()]
        extra_colours = [key for key, val in CSS4_COLORS.items()]
        colours += extra_colours[::-1]
        markers = [marker for marker in list(Line2D.markers.values())
                   if not marker in ['pixel', 'circle', 'nothing']]

        categories = []
        if subclasses is None:
            categories = ['{}_'.format(cat) for cat in sorted(list(set(list(y))))]
        else:
            categories = sorted(list(set(list(subclasses))))

        cat_class_colours = {}
        colour_count = 0
        for cat in categories:
            if cat.count('_') != 1:
                raise ValueError(
                    'Character "_" found in subclass {} more/less than '
                    'once'.format(cat)
                )
            cat_class = cat.split('_')[0]
            if not cat_class in cat_class_colours:
                cat_class_colours[cat_class] = colours[colour_count]
                colour_count += 1

        cat_markers = {}
        for cat_1_class in cat_class_colours.keys():
            marker_count = 0
            for cat_2 in categories:
                cat_2_class = cat_2.split('_')[0]
                if cat_1_class == cat_2_class:
                    cat_markers[cat_2.rstrip('_')] = markers[marker_count]
                    marker_count += 1

        cat_colours = {}
        for cat in cat_markers.keys():
            cat_class = cat.split('_')
            colour = cat_class_colours[cat_class]
            cat_colours[cat.rstrip('_')] = colour

        # Runs PCA
        pca = PCA(n_components=num_dimensions)
        pca_fitted = pca.fit(X=x)
        X_reduced = pca_fitted.transform(X=x)

        # Generates scatter plot
        fig = plt.figure()
        if num_dimensions == 2:
            ax = fig.add_subplot(111)
            for i, y_val in np.ndenumerate(y):
                scatter = ax.scatter(
                    X_reduced[i[0],0], X_reduced[i[0],1], c=cat_colours[y_val],
                    marker=cat_markers[y_val]
                )
        elif dimensions == 3:
            ax = fig.add_subplot(111, projection='3d')
            for i, y_val in np.ndenumerate(y):
                scatter = ax.scatter(
                    X_reduced[i[0],0], X_reduced[i[0],1], X_reduced[i[0],2],
                    c=cat_colours[y_val], marker=cat_markers[y_val]
                )

        legend_elements = []
        for cat, colour in cat_colours.items():
            marker = cat_markers[cat]
            legend_elements.append(
                Line2D([0], [0], marker=marker, color=colour, label=cat,
                markerfacecolor=colour)
            )
        ax.legend(handles=legend_elements, loc='upper right', title='Classes')

        plt.savefig('{}/{}_{}_PCA_plot.svg'.format(
            self.results_dir, plt_name, str(num_dimensions)
        ))
        if test is False:
            plt.show()

    def define_fixed_model_params(self, classifier):
        """
        For the 6 default ML algorithms run by this code (LogisticRegression,
        KNeighborsClassifier, GaussianNB, LinearSVC, SVC and
        AdaBoostClassifier), defines a dictionary of hyperparameters and
        their corresponding values that will remain fixed throughout
        optimisation and training

        Input
        ----------
        - classifier: The selected ML algorithm,
        e.g. sklearn.ensemble.AdaBoostClassifier()

        Output
        ----------
        - params: Dictionary of fixed hyperparameters
        """

        if type(classifier).__name__ == 'LogisticRegression':
            params = OrderedDict({'n_jobs': -1})
        elif type(classifier).__name__ == 'KNeighborsClassifier':
            params = OrderedDict({'metric': 'minkowski',
                                  'n_jobs': -1})
        elif type(classifier).__name__ == 'LinearSVC':
            params = OrderedDict({'dual': False})  # Change back to True
            # (= default) if n_samples < n_features
        elif type(classifier).__name__ == 'SVC':
            params = OrderedDict()
        elif type(classifier).__name__ == 'AdaBoostClassifier':
            params = OrderedDict()
        elif type(classifier).__name__ == 'GaussianNB':
            params = OrderedDict()

        return params

    def define_tuned_model_params(self, classifier, x_train, n_folds=5):
        """
        For the 6 default ML algorithms run by this code (LogisticRegression,
        KNeighborsClassifier, GaussianNB, LinearSVC, SVC and
        AdaBoostClassifier), returns dictionary of a sensible range of
        values for variable hyperparameters to be tested in randomised / grid
        search

        Input
        ----------
        - classifier: The selected ML algorithm,
        e.g. sklearn.ensemble.AdaBoostClassifier()
        - x_train: Numpy array of x values of training data
        - n_folds: The number of folds to be used in (stratified) k-folds
        cross-validation. By default set to 5

        Output
        ----------
        - params: Dictionary of parameter ranges that can be fed into
        RandomizedSearchCV or GridSearchCV
        """

        if type(n_folds) != int:
            raise TypeError(
                '"n_folds" should be set to a positive integer value'
            )
        else:
            if n_folds < 1:
                raise ValueError(
                    '"n_folds" should be set to a positive integer value'
                )

        shape = x_train.shape[0]
        if type(classifier).__name__ == 'LogisticRegression':
            params = OrderedDict({
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'sag', 'saga', 'newton-cg', 'lbfgs'],
                'multi_class': ['ovr', 'multinomial'],
                'C': np.logspace(-3, 5, 17)
            })
        elif type(classifier).__name__ == 'KNeighborsClassifier':
            if (1/n_folds)*shape < 2:
                raise AlgorithmError(
                    'Too few data points in dataset to run k nearest neighbours'
                )
            else:
                neighbours = np.array(range(2, int((1/n_folds)*shape), 1))
                params = OrderedDict({
                    'n_neighbors': neighbours,
                    'weights': ['uniform', 'distance'],
                    'p': np.array([1, 2])
                })
        elif type(classifier).__name__ == 'LinearSVC':
            params = OrderedDict({'C': np.logspace(-5, 15, num=41, base=2)})
        elif type(classifier).__name__ == 'SVC':
            # For speed reasons (some kernels take a prohibitively long time to
            # train) am sticking with the default kernel ('rbf')
            params = OrderedDict({
                'C': np.logspace(-5, 15, num=41, base=2),
                'gamma': np.logspace(-15, 3, num=37, base=2),
                'kernel': ['rbf']
            })
        elif type(classifier).__name__ == 'AdaBoostClassifier':
            if (1/n_folds)*shape < 2:
                raise AlgorithmError(
                    'Too few data points in dataset to use AdaBoost classifier'
                )
            else:
                n_estimators = [int(x) for x in np.logspace(1, 4, 7)]
                params = OrderedDict({'n_estimators': n_estimators})
        elif type(classifier).__name__ == 'GaussianNB':
            params = OrderedDict()

        return params

    def flag_extreme_params(self, best_params, poss_params, test=False):
        """
        Flags a warning if hyperparameter optimisation selects a value at the
        extreme end of an input (numerical) range

        Input
        ----------
        - best_params: Dictionary of 'optimal' parameters returned by
        hyperparameter optimisation algorithm (e.g. RandomizedSearchCV or
        GridSearchCV)
        - poss_params: Dictionary of hyperparameter ranges fed into the
        hyperparameter optimisation algorithm, such as that returned by
        define_model_params
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False
        """

        for param, best_val in best_params.items():
            poss_vals = poss_params[param]
            # If clauses are at separate indentations to avoid raising errors
            if isinstance(poss_vals, (list, np.ndarray)):
                if (
                        all(isinstance(poss_val, (int, float)) for poss_val in poss_vals)
                    and len(poss_vals) > 2
                ):
                    if best_val in [poss_vals[0], poss_vals[-1]]:
                        print('\x1b[31m WARNING: Optimal value selected for {} is at '
                              'the extreme of the range tested \033[0m'.format(param))
                        print('Range tested: {}'.format(poss_vals))
                        print('Value selected: {}'.format(best_val))
                        print('\n\n')

    def conv_resampling_method(self, resampling_method):
        """
        Converts resampling method name (a string) into an object

        Input
        ----------
        resampling_method: Name of resampling method (string)

        Output
        ----------
        resampling_obj: Resampling method (object)
        """

        if resampling_method == 'no_balancing':
            resampling_obj = None
        elif resampling_method == 'max_sampling':
            resampling_obj = RandomOverSampler(sampling_strategy='not majority')
        elif resampling_method == 'smote':
            resampling_obj = SMOTE(sampling_strategy='not majority')
        elif resampling_method == 'smoteenn':
            resampling_obj = SMOTEENN(sampling_strategy='not majority')
        elif resampling_method == 'smotetomek':
            resampling_obj = SMOTETomek(sampling_strategy='not majority')
        else:
            raise ValueError(
                'Resampling method {} not recognised'.format(resampling_method)
            )

        return resampling_obj

    def run_randomised_search(
        self, x_train, y_train, train_groups, selected_features, clf, splits,
        resampling_method, n_components_pca, params, scoring_func, n_iter='',
        test=False
    ):
        """
        Randomly picks combinations of hyperparameters from an input grid and
        runs cross-validation to calculate their score using the
        RandomizedSearchCV class from sklearn. The cross-validation pipeline
        consists of: 1) standardisation of the data across each feature (i.e.
        subtracting the mean and dividing by the standard deviation); 2)
        transforming the data to a user-specified number of features with PCA;
        3) resampling the data if imbalanced; and 4) running the selected
        classifier with randomly selected combinations of input hyperparameter
        values. Note that incompatible combinations of hyperparameters will
        simply be skipped (error_score=np.nan), so there is no need to avoid
        including e.g. hyperparameters that are defined only if other
        hyperparameters are defined as particular values. Returns the resuts of
        the randomised search.

        Input
        ----------
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - train_groups: Numpy array of group names of training data
        - selected_features:
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
        - splits: Generator function that yields train: test splits of the input
        data (x_train and y_train) for cross-validation
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to transform the data to after
        fitting the data with PCA. If set to None, PCA will not be included in
        the pipeline.
        - params: Dictionary of hyperparameter values to search.
        Key = hyperparameter name (must match the name of the parameter in the
        selected sklearn ML classifier class); Value = range of values to
        test for that hyperparameter - note that all numerical ranges must be
        supplied as numpy arrays in order to avoid throwing an error with the
        imblearn Pipeline() class
        - scoring_func: The function used to score the fitted classifier on the
        data set aside for validation during cross-validation. Either a
        function, or the name of one of the scoring functions in sklearn (see
        list of recognised names at https://scikit-learn.org/stable/modules/
        model_evaluation.html#scoring-parameter).
        - n_iter: The number of hyperparameter combinations to test, if set to ''
        will be set to either 25 or 10% of the total number of possible
        combinations of hyperparameter values specified in the params dictionary
        (whichever is larger).
        - test: Boolean describing whether the function is being run during the
        program's unit tests - by default is set to False

        Output
        ----------
        - random_search: RandomizedSearchCV object fitted to the training data
        """

        from imblearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.model_selection import RandomizedSearchCV

        # Determines number of iterations to run. If this is not defined by the
        # user, it is set to 1/3 * the total number of possible parameter value
        # combinations, with a lower limit of 100. If the total number of
        # possible parameter value combinations is less than 100, all parameter
        # value combinations are tested.
        num_params_combs = 1
        for val in params.values():
            if isinstance(val, (list, np.ndarray)):
                num_params_combs *= len(val)
        if n_iter == '':
            n_iter = int(num_params_combs*(1/3))
            if n_iter < 100:
                n_iter = 100
        if n_iter > num_params_combs:
            n_iter = num_params_combs

        # Generates resampling object
        resampling_obj = self.conv_resampling_method(resampling_method=resampling_method)

        # Runs randomised search pipeline
        feature_selection = ManualFeatureSelection(
            all_features=self.features, selected_features=selected_features
        )
        standardisation = StandardScaler()
        if n_components_pca is None:
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])
        else:
            pca = PCA(n_components=n_components_pca)
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('PCA', pca),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])

        params = OrderedDict({'{}__{}'.format(type(clf).__name__, key): val
                              for key, val in params.items()})

        random_search = RandomizedSearchCV(
            estimator=std_pca_clf, param_distributions=params, n_iter=n_iter,
            scoring=scoring_func, n_jobs=-1, cv=splits, error_score=np.nan
        )
        random_search.fit(X=x_train, y=y_train, groups=train_groups)

        self.flag_extreme_params(random_search.best_params_, params)

        return random_search

    def run_grid_search(
        self, x_train, y_train, train_groups, selected_features, clf, splits,
        resampling_method, n_components_pca, params, scoring_func
    ):
        """
        Tests all possible combinations of hyperparameters from an input grid
        and runs cross-validation to calculate their score using the
        GridSearchCV class from sklearn. The cross-validation pipeline
        consists of: 1) standardisation of the data across each feature (i.e.
        subtracting the mean and dividing by the standard deviation); 2)
        transforming the data to a user-specified number of features with PCA;
        3) resampling the data if imbalanced; and 4) running the selected
        classifier with all possible combinations of input hyperparameter
        values. Note that incompatible combinations of hyperparameters will
        simply be skipped (error_score=np.nan), so there is no need to avoid
        including e.g. hyperparameters that are defined only if other
        hyperparameters are defined as particular values. Returns the results of
        the grid search.

        Input
        ----------
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - train_groups: Numpy array of group names of training data
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
        - splits: Generator function that yields train: test splits of the input
        data (x_train and y_train) for cross-validation
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to transform the data to after
        fitting the data with PCA. If set to None, PCA will not be included in
        the pipeline.
        - params: Dictionary of hyperparameter values to search.
        Key = hyperparameter name (must match the name of the parameter in the
        selected sklearn ML classifier class); Value = range of values to
        test for that hyperparameter - note that all numerical ranges must be
        supplied as numpy arrays in order to avoid throwing an error with the
        imblearn Pipeline() class
        - scoring_func: The function used to score the fitted classifier on the
        data set aside for validation during cross-validation. Either a
        function, or the name of one of the scoring functions in sklearn (see
        list of recognised names at https://scikit-learn.org/stable/modules/
        model_evaluation.html#scoring-parameter).

        Output
        ----------
        - grid_search: GridSearchCV object fitted to the training data
        """

        from imblearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.model_selection import GridSearchCV

        # Generates resampling object
        resampling_obj = self.conv_resampling_method(resampling_method=resampling_method)

        feature_selection = ManualFeatureSelection(
            all_features=self.features, selected_features=selected_features
        )
        standardisation = StandardScaler()
        if n_components_pca is None:
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])
        else:
            pca = PCA(n_components=n_components_pca)
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('PCA', pca),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])

        params = OrderedDict({'{}__{}'.format(type(clf).__name__, key): val
                              for key, val in params.items()})

        grid_search = GridSearchCV(
            estimator=std_pca_clf, param_grid=params, scoring=scoring_func,
            n_jobs=-1, cv=splits, error_score=np.nan
        )
        grid_search.fit(X=x_train, y=y_train, groups=train_groups)

        self.flag_extreme_params(grid_search.best_params_, params)

        return grid_search

    def train_model(
        self, x_train, y_train, train_groups, selected_features, clf,
        resampling_method, n_components_pca, scoring_func
    ):
        """
        Trains user-specified model on the training data (without cross-
        validation). Training data is first standardised (by subtracting the
        mean) and dividing by the standard deviation, then transformed to a
        user-specified number of features with PCA, and finally resampled if
        necessary to balance the class sizes, before it is fed into the model
        for training.

        Input
        ----------
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - train_groups: Numpy array of group names of training data
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to transform the data to
        after fitting the data with PCA. If set to None, PCA will not be
        included in the pipeline.
        - scoring_func: The function used to score the fitted classifier on the
        data set aside for validation during cross-validation. Either a
        function, or the name of one of the scoring functions in sklearn (see
        list of recognised names at https://scikit-learn.org/stable/modules/
        model_evaluation.html#scoring-parameter).

        Output
        ----------
        - std_pca_clf: Model fitted to the training data. This can be fed into
        test_model to measure how well the model predicts the classes of the
        test data set aside by split_train_test_data.
        """

        from imblearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        from sklearn.decomposition import PCA
        from sklearn.model_selection import cross_val_score

        # Generates resampling object
        resampling_obj = self.conv_resampling_method(resampling_method=resampling_method)

        feature_selection = ManualFeatureSelection(
            all_features=self.features, selected_features=selected_features
        )
        standardisation = RobustScaler()
        if n_components_pca is None:
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])
        else:
            pca = PCA(n_components=n_components_pca)
            std_pca_clf = Pipeline([('feature_selection', feature_selection),
                                    ('std', standardisation),
                                    ('PCA', pca),
                                    ('resampling', resampling_obj),
                                    (type(clf).__name__, clf)])

        # Fits single model on all training data
        std_pca_clf.fit(X=x_train, y=y_train)

        return std_pca_clf

    def test_model(
        self, x_test, y_test, clf, test_scoring_funcs, draw_conf_mat=True
    ):
        """
        Tests model (previously fitted to the training data using e.g.
        train_model function) by predicting the class labels of test data.
        Scores the model by comparing the predicted and actual class labels
        across a range of user-specified scoring functions, and plots confusion
        matrix.

        Input
        ----------
        - x_test: Numpy array of x values of test data
        - y_test: Numpy array of y values of test data
        - clf: Model previously fitted to the training data
        - test_scoring_funcs: Dictionary of sklearn scoring functions and
        dictionaries of arguments to be fed into these functions.
        E.g. sklearn.metrics.f1_score: {'average': 'macro'}
        - draw_conf_mat: Boolean, dictates whether or not confusion matrices are
        plotted.

        Output
        ----------
        - test_scores: Dictionary of user-specified scoring functions and their
        values as calculated from the class label predictions made by the model
        on the testing data
        """

        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels

        # Uses trained model to predict classes of test data. Note that
        # standardisation and PCA will be performed within the pipeline, so
        # don't need to do this manually.
        predictions = clf.predict(X=x_test)

        # Calculates selected scoring metrics
        test_scores = OrderedDict()
        for func, params in test_scoring_funcs.items():
            if func.__name__ == 'cohen_kappa_score':
                params['y1'] = y_test
                params['y2'] = predictions
            else:
                params['y_true'] = y_test
                params['y_pred'] = predictions
            test_score = func(**params)
            test_scores[func.__name__.replace('_score', '')] = test_score


        # Generates confusion matrices
        if draw_conf_mat is True:
            normalisation_methods = OrderedDict({None: '',
                                                 'true': ['_recall', 'rows'],
                                                 'pred': ['_precision', 'columns']})
            for method, method_label in normalisation_methods.items():
                if method is not None:
                    print('Normalised over {} label ({})'.format(
                        method, method_label[1]
                    ))
                plt.clf()
                labels = unique_labels(y_test, predictions)
                # Below ensures that predicted and true labels are on the correct axes,
                # so think carefully before updating!
                sns.heatmap(
                    data=confusion_matrix(y_true=y_test, y_pred=predictions,
                                          labels=labels, normalize=method),
                    cmap='RdBu_r', annot=True, xticklabels=True,
                    yticklabels=True, fmt='.3f'
                )
                ax = plt.gca()
                ax.set(xticklabels=labels, yticklabels=labels, xlabel='Predicted label',
                       ylabel='True label')
                plt.xticks(rotation='vertical')
                plt.yticks(rotation='horizontal')
                plt.savefig('{}/{}{}_confusion_matrix.svg'.format(
                    self.results_dir, type(clf).__name__, method_label[0]
                ))
                plt.show()

        return predictions, test_scores

    def run_ml(
        self, clf, x_train, y_train, train_groups, x_test, y_test,
        selected_features, n_components_pca, run, params, train_scoring_func,
        test_scoring_funcs=None, resampling_method='no_balancing', n_iter='',
        cv_folds=5, draw_conf_mat=True
    ):
        """
        Loops over user-specified data resampling methods and runs clf algorithm
        to: either a) test a range of hyperparameter combinations with
        RandomizedSearchCV or GridSearchCV to decide upon the optimal
        combination of hyperparameter values; or b) train and test the model.

        Input
        ----------
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - train_groups: Numpy array of group names of training data
        - x_test: Numpy array of x values of test data
        - y_test: Numpy array of y values of test data
        - n_components_pca: The number of components to transform the data to
        after fitting the data with PCA. If set to None, PCA will not be
        included in the pipeline.
        - run: Either 'randomsearch', 'gridsearch' or 'train'. Directs the
        function whether to run cross-validation with RandomizedSearchCV or
        GridSearchCV to select a suitable combination of hyperparameter values,
        or whether to train and test the model.
        - params: If run == 'randomsearch' or 'gridsearch', params is a
        dictionary of hyperparameter values to search. Key = hyperparameter name
        (must match the name of the parameter in the selected sklearn ML
        classifier class); Value = range of values to test for that
        hyperparameter - note that all numerical ranges must be supplied as
        numpy arrays in order to avoid throwing an error with the imblearn
        Pipeline() class. Else if run == 'train', params is also a dictionary of
        hyperparameters, but in this case a single value is provided for each
        hyperparameter as opposed to a range.
        - train_scoring_func: The function used to score the fitted classifier
        on the data set aside for validation during cross-validation. Either a
        function, or the name of one of the scoring functions in sklearn (see
        list of recognised names at https://scikit-learn.org/stable/modules/
        model_evaluation.html#scoring-parameter).
        - test_scoring_funcs: Dictionary of sklearn scoring functions and
        dictionaries of arguments to be fed into these functions.
        E.g. sklearn.metrics.f1_score: {'average': 'macro'}
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_iter: Integer number of hyperparameter combinations to test / ''. If
        set to '' will be set to either 100 or 1/3 of the total number of
        possible combinations of hyperparameter values specified in the params
        dictionary (whichever is larger).
        - cv_folds: Integer number of folds to run in cross-validation. E.g. as
        a default cv_folds = 5, which generates 5 folds of training and
        validation data, with 80% and 20% of the data forming the training and
        validation sets respectively.
        - draw_conf_mat: Boolean, dictates whether to plot confusion matrices to
        compare the model predictions to the test data

        Output
        ----------
        - searches: Dictionary of the selected resampling methods and the
        corresponding output from fitting the user-specified algorithm (either a
        RandomizedSearchCV / GridSearchCV object, or a model fitted to the
        resampled training data).
        - train_scores: The value of the selected scoring function calculated
        from cross-validation of the model on the training data.
        - test_scores: Dictionary of user-specified scoring functions and their
        values as calculated from the class label predictions made by the model
        on the testing data
        """

        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import GroupKFold

        if run in ['randomsearch', 'gridsearch']:
            skf = ''
            gkf = ''
            if self.randomise is True:  No longer defined
                # There must be more than cv_folds instances of each dataset
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True)
                splits = skf.split(X=x_train, y=y_train)
            else:
                gkf = GroupKFold(n_splits=cv_folds)
                splits = gkf.split(X=x_train, y=y_train, groups=train_groups)
        else:
            splits = ''

        if run == 'randomsearch':
            search = self.run_randomised_search(
                x_train, y_train, train_groups, selected_features, clf, splits,
                resampling_method, n_components_pca, params, train_scoring_func,
                n_iter
            )
            return search
        elif run == 'gridsearch':
            search = self.run_grid_search(
                x_train, y_train, train_groups, selected_features, clf, splits,
                resampling_method, n_components_pca, params, train_scoring_func
            )
            return search
        elif run == 'train':
            search = self.train_model(
                x_train, y_train, train_groups, selected_features, clf,
                resampling_method, n_components_pca, train_scoring_func
            )
            predictions, test_scores = self.test_model(
                x_test, y_test, search, test_scoring_funcs, draw_conf_mat
            )
            return search, test_scores, predictions

    def run_nested_CV(
        self, clf, x, y, groups, selected_features, n_components_pca, run,
        fixed_params, tuned_params, train_scoring_func,
        test_scoring_funcs=None, resampling_method='no_balancing', n_iter='',
        cv_folds_inner_loop=5, cv_folds_outer_loop='loocv', draw_conf_mat=False
    ):
        """
        Fits an input sklearn classifier to the data.

        Input
        ----------
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.ensemble.RandomForestClassifier
        - x: Numpy array of all x data (no need to have already split into
        training and test data)
        - y: Numpy array of all y data (no need to have already split into
        training and test data)
        - groups: Numpy array of all group names (no need to have already split
        into training and test data)
        - n_components_pca: The number of components to transform the data to
        after fitting the data with PCA. If set to None, PCA will not be
        included in the pipeline.
        - run: Either 'randomsearch' or 'gridsearch'. Directs the function
        whether to run inner loop cross-validation with RandomizedSearchCV or
        GridSearchCV to select a suitable combination of hyperparameter values.
        - fixed_params: Dictionary of hyperparameters and their selected values
        that remain constant regardless of the value of 'run' (e.g. n_jobs = -1)
        - tuned_params: Dictionary of hyperparameter values to search.
        Key = hyperparameter name (must match the name of the parameter in the
        selected clf class); Value = range of values to test for that
        hyperparameter - note that all numerical ranges must be supplied as
        numpy arrays in order to avoid throwing an error with the imblearn
        Pipeline() class.
        - train_scoring_func: The function used to score the fitted classifier
        on the data set aside for validation during cross-validation. Either a
        function, or the name of one of the scoring functions in sklearn (see
        list of recognised names at https://scikit-learn.org/stable/modules/
        model_evaluation.html#scoring-parameter).
        - test_scoring_funcs: Dictionary of sklearn scoring functions and
        dictionaries of arguments to be fed into these functions.
        E.g. sklearn.metrics.f1_score: {'average': 'macro'}
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'smoteenn'; and 'smotetomek'.
        - n_iter: Required if run == 'randomsearch'. Integer number of
        hyperparameter combinations to test / ''. If set to '' will be set to
        either 100 or 1/3 of the total number of possible combinations of
        hyperparameter values specified in the params dictionary (whichever is
        larger).
        - cv_folds_inner_loop: Integer number of folds to run in
        cross-validation (loops = 1) / the inner cross-validation loop
        (loops = 2). E.g. as a default cv_folds_inner_loop = 5, which generates
        5 folds of training and validation data, with 80% and 20% of the data
        forming the  training and validation sets respectively.
        - cv_folds_outer_loop: Integer number of folds to run in the outer
        cross-validation loop (loops = 2). Set to 'loocv' as default, which
        specifies a number of folds equal to the size of the dataset
        (leave-one-out cross-validation)
        - draw_conf_mat: Boolean, dictates whether to plot confusion matrices to
        compare the model predictions to the test data

        Output
        ----------
        - searches: Dictionary of the selected resampling methods and the
        corresponding output from fitting the user-specified algorithm (either a
        RandomizedSearchCV / GridSearchCV object, or a model fitted to the
        resampled training data).
        - train_scores: The value of the selected scoring function calculated
        from cross-validation of the model on the training data.
        - test_scores: Dictionary of user-specified scoring functions and their
        values as calculated from the class label predictions made by the model
        on the testing data
        """

        from sklearn.model_selection import (
            GroupKFold, LeaveOneOut, StratifiedKFold
        )

        nested_cv_search = OrderedDict({
            'inner_loop_searches': [],
            'outer_loop_models': [],
            'outer_loop_params': [],
            'test_scores': {},
            'predictions': [],
            'x_true': [],
            'y_true': []
        })
        for scoring_func in test_scoring_funcs.keys():
            nested_cv_search['test_scores'][
                scoring_func.__name__.replace('_score', '')
            ] = []

        # Split data into train and test sets
        loocv = ''
        skf = ''
        gkf = ''
        if type(cv_folds_outer_loop) == str:
            if cv_folds_outer_loop.lower().replace(' ', '') == 'loocv':
                loocv = LeaveOneOut()
                splits = loocv.split(X=x, y=y)
            else:
                raise ValueError(
                    'Value {} for CV method in outer loop not recognised - '
                    'set to either "loocv" or an integer number of '
                    'folds'.format(cv_folds_outer_loop)
                )
        else:
            if self.randomise is True:  No longer defined
                skf = StratifiedKFold(n_splits=cv_folds_outer_loop, shuffle=True)
                splits = skf.split(X=x, y=y)
            else:
                gkf = GroupKFold(n_splits=cv_folds_outer_loop)
                splits = gkf.split(X=x, y=y, groups=groups)

        for split in splits:
            train_split = split[0]
            test_split = split[1]
            x_train = x[train_split]
            y_train = y[train_split]
            x_test = x[test_split]
            y_test = y[test_split]
            if groups is not None:
                train_groups = groups[train_split]
            else:
                train_groups = None

            # Random / grid search of hyperparameters on training set
            train_clf = clf(**fixed_params)
            search = self.run_ml(
                train_clf, x_train, y_train, train_groups, '', '',
                selected_features, n_components_pca, run, tuned_params,
                train_scoring_func, {}, resampling_method, n_iter,
                cv_folds_inner_loop, draw_conf_mat
            )
            nested_cv_search['inner_loop_searches'].append(search)
            best_params = OrderedDict({
                key.split('__')[1]: val for key, val in search.best_params_.items()
            })
            # Train clf with best hyperparameter selection on training split,
            # then make predictions and calculate selected statistics on test
            # split
            best_params = OrderedDict({**fixed_params, **best_params})
            test_clf = clf(**best_params)
            split_search, split_test_scores, split_predictions = self.run_ml(
                test_clf, x_train, y_train, train_groups, x_test, y_test,
                selected_features, n_components_pca, 'train', {},
                train_scoring_func, test_scoring_funcs, resampling_method,
                n_iter, '', draw_conf_mat
            )
            nested_cv_search['outer_loop_params'].append(best_params)
            nested_cv_search['outer_loop_models'].append(split_search)

            for scoring_func_name, val in split_test_scores.items():
                nested_cv_search['test_scores'][scoring_func_name].append(val)
            nested_cv_search['predictions'].append(split_predictions)
            nested_cv_search['x_true'].append(x_test)
            nested_cv_search['y_true'].append(y_test)

        # Calculate average, standard deviation and percentiles of interest
        # across cv_folds_outer_loop folds
        nested_cv_search['average_test_scores'] = OrderedDict()
        for scoring_func_name, score_list in nested_cv_search['test_scores'].items():
            nested_cv_search['average_test_scores'][scoring_func_name] = np.mean(score_list)
        nested_cv_search['std_test_scores'] = OrderedDict()
        for scoring_func_name, score_list in nested_cv_search['test_scores'].items():
            nested_cv_search['std_test_scores'][scoring_func_name] = np.std(score_list, ddof=0)  # Population standard deviation
        nested_cv_search['percentile_test_scores'] = OrderedDict()
        for scoring_func_name, score_list in nested_cv_search['test_scores'].items():
            nested_cv_search['percentile_test_scores'][scoring_func_name] = [
                np.percentile(score_list, 2.5), np.percentile(score_list, 50),
                np.percentile(score_list, 97.5)
            ]
        best_index = np.where(
               nested_cv_search['test_scores'][train_scoring_func]
            == np.amax(nested_cv_search['test_scores'][train_scoring_func])
        )[0][0]
        nested_cv_search['best_outer_loop_params'] = nested_cv_search['outer_loop_params'][best_index]

        return nested_cv_search

    def run_5x2_CV_paired_t_test(
        self, x, y, groups, selected_features_1, selected_features_2,
        classifier_1, classifier_2, params_1, params_2, resampling_method_1,
        resampling_method_2, n_components_pca_1, n_components_pca_2, scoring_func
    ):
        """
        Runs 5x2 CV combined F test to calculate whether there is a significant
        difference in performance between two classifier models.
        """

        from mlxtend.evaluate import combined_ftest_5x2cv

        clf_1 = classifier_1(**params_1)
        trained_clf_1 = self.train_model(
            x, y, groups, selected_features_1, clf_1, resampling_method_1,
            n_components_pca_1, scoring_func
        )
        clf_2 = classifier_2(**params_2)
        trained_clf_2 = self.train_model(
            x, y, groups, selected_features_2, clf_2, resampling_method_2,
            n_components_pca_2, scoring_func
        )

        F, p = combined_ftest_5x2cv(
            estimator1=trained_clf_1, estimator2=trained_clf_2, X=x, y=y
        )

        return F, p
