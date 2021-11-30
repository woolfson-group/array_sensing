

# Functions to perform ML analysis on parsed BADASS data.

import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin

sns.set()

if __name__ == 'subroutines.train':
    from subroutines.parse_array_data import DefData
    from subroutines.exceptions import AlgorithmError
else:
    from array_sensing.subroutines.parse_array_data import DefData
    from array_sensing.subroutines.exceptions import AlgorithmError

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
        - fluor_data: DataFrame of (standardised) fluorescence readings (output
        from parse_array_data class)
        - classes: Numpy array / list defining class labels associated with the
        samples, e.g. for analysis of tea types an example list would be
        ['Green', 'Green', 'Black', 'Black', 'Grey', 'Grey'].
        If not defined, code will select the column labelled 'Analyte' in the
        fluor_data DataFrame.
        - subclasses: Numpy array / list defining subclass labels associated
        with the samples (if such subclass labels exist), e.g. for analysis of
        tea types an example list would be ['Green_Tetleys', 'Green_Pukka',
        'Black_Yorkshire', 'Black_Yorkshire', 'Grey_Asda', 'Grey_Pukka']
        - shuffle: Boolean. If set to True, will randomly shuffle the input data
        """

        DefData.__init__(self, results_dir)

        # Checks that if subclasses are defined, classes are also defined
        self.classes = classes
        if not subclasses is None and self.classes is None:
            raise TypeError(
                '"subclasses" argument cannot be set to a value other than None'
                ' if the "classes" argument is set equal to None'
            )

        # If subclasses are defined, assumes ML analysis will focus upon ability
        # to extrapolate trained model to new subclasses
        if subclasses is None:
            self.subclass_split = False
        else:
            self.subclass_split = True

        # Ensures that no columns in the input dataframe will be overwritten
        # when 'Classes' and 'Subclasses' columns are subsequently appended
        if any(x in fluor_data.columns for x in ['Classes', 'Subclasses']):
            raise NameError(
                'Please rename any columns in input dataframe labelled either '
                '"Classes" or "Subclasses", as these columns are added to the '
                'dataframe by the code during data processing'
            )

        class_dict = OrderedDict({'Classes': self.classes,
                                  'Subclasses': subclasses})
        fluor_data = fluor_data.reset_index(drop=True)

        # Checks that, if defined, classes and subclasses arguments are numpy
        # arrays, and randomly shuffles the data if shuffle set to True.
        for class_name, class_list in class_dict.items():
            error = False

            if class_list is None:
                continue

            if isinstance(class_list, np.ndarray):
                if len(class_list.shape) != 1:
                    error = True
            elif isinstance(class_list, list):
                pass
            else:
                error = True

            if error is True:
                raise TypeError(
                    '"classes" and "subclasses" arguments should be either a 1d'
                    ' numpy array or a list'
                )
            else:
                class_data = pd.DataFrame({class_name: class_list})
                fluor_data = pd.concat([fluor_data, class_data], axis=1)

        # Randomly shuffles input data
        if shuffle is True:
            indices = [i for i in range(fluor_data.shape[0])]
            random.shuffle(indices)
            fluor_data = fluor_data.iloc[indices].reset_index(drop=True)
        if not self.classes is None:
            self.classes = fluor_data['Classes'].to_numpy()
            fluor_data = fluor_data.drop(['Classes'], axis=1)
        if not subclasses is None:
            subclasses = fluor_data['Subclasses'].to_numpy()
            fluor_data = fluor_data.drop(['Subclasses'], axis=1)
        self.fluor_data = fluor_data

        # Defines 'x', 'y' and 'groups' variables for ML with sklearn
        self.drop_cols = [col for col in self.fluor_data.columns
                          if 'analyte' in col.lower()]
        self.features = self.fluor_data.drop(self.drop_cols, axis=1).columns.tolist()
        self.x = copy.deepcopy(self.fluor_data.drop(self.drop_cols, axis=1)).to_numpy()
        if not self.classes is None:
            self.y = copy.deepcopy(self.classes)
        else:
            self.y = copy.deepcopy(self.fluor_data['Analyte']).to_numpy()
        self.groups = subclasses

    def scramble_class_labels(self):
        """
        Randomly shuffles y (and, if defined, linked group) values relative to
        x values (which are left unshuffled)
        """

        index = [n for n in range(self.fluor_data.shape[0])]
        random.shuffle(index)

        for col in self.drop_cols:
            self.fluor_data[col] = self.fluor_data[col].iloc[index].tolist()
        self.y = self.y[index]
        if self.subclass_split is True:
            self.groups = self.groups[index]

    def randomly_pick_subclasses(self, one_of_each, percent_test):
        """
        Randomly selects subclasses to be saved as test set.

        Input
        --------
        - one_of_each: Boolean, determines whether test set includes one or more
        subclass of each class (True), or one or more subclass(es) selected at
        random (False). N.B. If set to True, number of subclasses in each class
        must be equal.
        - percent_test: The percentage of the data to be set aside as a test
        set. Must be a multiple of either the total number of subclasses
        (one_of_each = False), or the number of subclasses in each class
        (one_of_each = True)

        Output
        --------
        - test_classes: List of lists of subclasses selected to form the test
        set in successive train test splits
        """

        if self.subclass_split is False:
            raise TypeError(
                'Inappropriate data format for this function. You need to '
                'define the classes argument when initialising a run_ml object.'
            )
        if percent_test < 0 or percent_test > 1:
            raise ValueError('percent_test should be a float between 0 and 1')

        subclasses = list(self.groups)
        classes = list(self.y)
        set_subclasses = list(set(subclasses))
        set_classes = list(set(classes))

        # Random split
        if one_of_each is False:
            # Checks that percent_test is a multiple of the total number of
            # subclasses
            if len(set_subclasses) % (1/percent_test) != 0:
                raise ValueError(
                    'percent_test is not a multiple of the total number of '
                    'subclasses'
                )

            test_classes = []
            random_classes = copy.deepcopy(set_subclasses)
            random.shuffle(random_classes)
            frac = int(len(set_subclasses)*percent_test)
            for n in range(int(1/percent_test)):
                test_classes.append(random_classes[n*frac:(n+1)*frac])

        # Stratified class split
        elif one_of_each is True:
            # Checks for equal class sizes
            organised_classes = OrderedDict()
            for n in range(len(classes)):
                class_val = classes[n]
                subclass_val = subclasses[n]
                if not class_val in organised_classes:
                    organised_classes[class_val] = []
                if not subclass_val in organised_classes[class_val]:
                    organised_classes[class_val].append(subclass_val)

            class_lens = [len(organised_classes[class_val])
                          for class_val in organised_classes.keys()]

            if len(set(class_lens)) != 1:
                raise RuntimeError(
                    'Class sizes are unequal:\n{}\nPlease update your input '
                    'dataset to contain equal class sizes'
                )

            # Creates list of subclass train test splits
            for class_val in organised_classes.keys():
                random.shuffle(organised_classes[class_val])

            sub_class_array = np.array(list(organised_classes.values())).transpose()

            num_test = sub_class_array.shape[0]*percent_test
            if (sub_class_array.shape[0] / num_test) % 1 != 0:
                raise ValueError(
                    'Multiplying the class size by percent_test gives a '
                    'non-integer value.\nPlease change the value of '
                    'percent_test to allow division of the classes into subsets'
                    ' of equal and integer size'
                )
            else:
                num_test = int(num_test)  # range() requires integer values

            test_classes = [
                sub_class_array[i:i+num_test,].flatten('C').tolist() for i in
                [int(n) for n in range(0, sub_class_array.shape[0], num_test)]
            ]

        return test_classes

    def split_train_test_data(
        self, randomise, percent_test=None, test_analytes=None
    ):
        """
        Splits data into training and test set

        Input
        --------
        - randomise: If set to True, percent_test % data points will be
        selected at random to form the test set. If set to False, the test set
        will be generated from data points collected for the analyte(s) listed
        under the test_analytes argument
        - percent_test: If randomise is set to True, this argument specifies
        the percentage of the total data to be set aside for model testing
        - test_analytes: If randomise is set to False, this argument lists the
        names of the analytes to be set aside for model testing
        """

        from sklearn.model_selection import StratifiedKFold

        num_data_points = self.fluor_data.shape[0]

        self.randomise = randomise
        if self.randomise is True:
            if 0 < percent_test < 1:
                n_splits = round(1 / percent_test)
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
                splits = skf.split(X=self.x, y=self.y)
                for split in splits:
                    train_set = list(split[0])
                    test_set = list(split[1])
                    break
            elif percent_test == 0:
                train_set = list(range(0, self.x.shape[0]))
                test_set = []
            else:
                raise ValueError('percent_test lies outside of the expected '
                                 'range of 0 <= percent_test < 1')


        elif self.randomise is False:
            test_set = []
            train_set = []

            for n in range(num_data_points):
                if self.groups[n] in test_analytes:
                    test_set.append(n)
                else:
                    train_set.append(n)

        self.train_data = self.fluor_data.iloc[train_set].reset_index(drop=True)
        self.test_data = self.fluor_data.iloc[test_set].reset_index(drop=True)

        self.train_x = self.x[train_set]
        self.test_x = self.x[test_set]
        self.train_y = self.y[train_set]
        self.test_y = self.y[test_set]
        if self.randomise is True:
            self.train_groups = None
            self.test_groups = None
        elif self.randomise is False:
            self.train_groups = self.groups[train_set]
            self.test_groups = self.groups[test_set]

    def calc_feature_correlations(self, train_data, plt_name='', abs_vals=True):
        """
        Calculates pairwise Kendall tau rank correlation coefficient values
        between all 2-way combinations of features, and plots a heatmap.

        Input
        --------
        - train_data: DataFrame of the training data, with features as column
        names
        - abs_vals: Colour heatmap by absolute rather than raw correlation
        coefficient values (to avoid strong negative correlations appearing to
        be weakly correlated when glancing at the heatmap)

        Output
        --------
        - correlation matrix: DataFrame of Kendall tau rank correlation
        coefficient values for all pairwise combinations of features
        """

        feature_corr_df = train_data.drop(self.drop_cols, axis=1)  # Must be a
        # dataframe, not a numpy array
        correlation_matrix = feature_corr_df.corr(method='kendall')
        if abs_vals is True:
            correlation_matrix = correlation_matrix.abs()

        plt.clf()
        heatmap = sns.heatmap(
            data=correlation_matrix, cmap='RdBu_r', annot=True,
            xticklabels=True, yticklabels=True, fmt='.3f'
        )
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        plt.savefig(
            '{}/{}Feature_correlations_Kendall_tau_rank.svg'.format(
                self.results_dir, plt_name
            )
        )
        plt.show()

        return correlation_matrix

    def calc_feature_importances_kbest(
        self, x, y, features, method_classif, scale, plt_name=''
    ):
        """
        Runs a univariate statistical test (either f_classif or
        mutual_info_classif) between x and y to calculate the importances of the
        different input features to these scores, and plots these values.

        Input
        --------
        - x: Numpy array of x values
        - y: Numpy array of y values
        - method_classif: Either 'f_classif' or 'mutual_info_classif'. From the
        scikit-learn docs - 'The methods based on F-test (f_classif) estimate
        the degree of linear dependency between two random variables. On the
        other hand, mutual information methods can capture any kind of
        statistical dependency, but being nonparametric, they require more
        samples for accurate estimation.'

        Output
        --------
        - score_df: DataFrame of features and their scores
        """

        from sklearn.feature_selection import SelectKBest
        from sklearn.preprocessing import RobustScaler

        univ_feature_importances = OrderedDict()
        for col in features:
            univ_feature_importances[col] = [np.nan for n in range(100)]

        for n in range(100):
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

            model = SelectKBest(score_func=method_classif, k='all')
            model.fit(X=temp_x, y=temp_y)
            total_y = np.sum(model.scores_)
            norm_y = model.scores_ / total_y

            for col, importance in enumerate(norm_y):
                col = features[col]
                univ_feature_importances[col][n] = importance

        cols = []
        cols_all = []
        univ_all = []
        univ_median = []
        univ_lower_conf_limit = []
        univ_upper_conf_limit = []
        for col, importances in univ_feature_importances.items():
            cols.append(col)
            univ_median.append(np.median(importances))
            univ_lower_conf_limit.append(np.percentile(importances, 2.5))
            univ_upper_conf_limit.append(np.percentile(importances, 97.5))
            for importance in importances:
                cols_all.append(col)
                univ_all.append(importance)

        plt.clf()
        plt.figure(figsize=(15,6))
        sns.barplot(x=cols, y=univ_median)
        sns.stripplot(x=cols, y=univ_lower_conf_limit, edgecolor='k',
                      linewidth=1, s=6, jitter=False)
        sns.stripplot(x=cols, y=univ_upper_conf_limit, edgecolor='k',
                      linewidth=1, s=6, jitter=False)
        plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
        plt.xlabel('Barrel')
        plt.ylabel('Importance')
        plt.savefig('{}/{}KBest_feat_importance_percentiles.svg'.format(
            self.results_dir, plt_name
        ))
        plt.show()

        plt.clf()
        plt.figure(figsize=(15,6))
        sns.stripplot(x=cols_all, y=univ_all, edgecolor='k', linewidth=1,
                      size=2.5, alpha=0.2)
        plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
        plt.xlabel('Barrel')
        plt.ylabel('Importance')
        plt.savefig('{}/{}KBest_feat_importance_all_data.svg'.format(
            self.results_dir, plt_name
        ))
        plt.show()

        score_df = pd.DataFrame({'Feature': cols, 'Score': univ_median})
        score_df = score_df.sort_values(
            by=['Score'], axis=0, ascending=False
        ).reset_index(drop=True)

        return score_df

    def calc_feature_importances_tree(self, x, y, features, scale, plt_name=''):
        """
        Input
        --------
        - x: Numpy array of x values
        - y: Numpy array of y values

        Output
        --------
        - feature_importances:
        """

        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.preprocessing import RobustScaler

        tree_feature_importances = OrderedDict()
        for col in features:
            tree_feature_importances[col] = [np.nan for n in range(100)]

        for n in range(100):
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

            model = ExtraTreesClassifier()
            model.fit(X=temp_x, y=temp_y)
            feature_importances = model.feature_importances_

            for col, importance in enumerate(model.feature_importances_):
                col = features[col]
                tree_feature_importances[col][n] = importance

        cols = []
        cols_all = []
        tree_all = []
        tree_median = []
        tree_lower_conf_limit = []
        tree_upper_conf_limit = []
        for col, importances in tree_feature_importances.items():
            cols.append(col)
            tree_median.append(np.median(importances))
            tree_lower_conf_limit.append(np.percentile(importances, 2.5))
            tree_upper_conf_limit.append(np.percentile(importances, 97.5))
            for importance in importances:
                cols_all.append(col)
                tree_all.append(importance)

        plt.clf()
        plt.figure(figsize=(15,6))
        sns.barplot(x=cols, y=tree_median)
        sns.stripplot(x=cols, y=tree_lower_conf_limit, edgecolor='k',
                      linewidth=1, s=6, jitter=False)
        sns.stripplot(x=cols, y=tree_upper_conf_limit, edgecolor='k',
                      linewidth=1, s=6, jitter=False)
        plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
        plt.xlabel('Barrel')
        plt.ylabel('Importance')
        plt.savefig('{}/{}Tree_feat_importance_percentiles.svg'.format(
            self.results_dir, plt_name
        ))
        plt.show()

        plt.clf()
        plt.figure(figsize=(15,6))
        sns.stripplot(x=cols_all, y=tree_all, edgecolor='k', linewidth=1,
                      size=2.5, alpha=0.2)
        plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
        plt.xlabel('Barrel')
        plt.ylabel('Importance')
        plt.savefig('{}/{}Tree_feat_importance_all_data.svg'.format(
            self.results_dir, plt_name
        ))
        plt.show()

        importance_df = pd.DataFrame({'Feature': cols,
                                      'Score': tree_median})
        importance_df = importance_df.sort_values(
            by=['Score'], axis=0, ascending=False
        ).reset_index(drop=True)

        return importance_df

    def calc_feature_importances_permutation(
        self, x, y, features, scale, plt_name=''
    ):
        """
        """

        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV

        permutation_feature_importances = OrderedDict()
        for col in features:
            permutation_feature_importances[col] = [np.nan for n in range(100)]

        for n in range(100):
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

            ada_boost = AdaBoostClassifier()
            parameters = OrderedDict({'n_estimators': [10, 50, 100, 500, 1000]})
            grid_search = GridSearchCV(
                estimator=ada_boost, param_grid=parameters, error_score=np.nan,
                scoring='accuracy'
            )
            grid_search.fit(X=x, y=y)

            ada_boost = AdaBoostClassifier(**grid_search.best_params_)
            ada_boost.fit(temp_x, temp_y)
            results = permutation_importance(
                ada_boost, temp_x, temp_y, scoring='accuracy', n_jobs=-1
            )

            for col, importance in enumerate(results.importances_mean):
                col = features[col]
                permutation_feature_importances[col][n] = importance

        cols = []
        cols_all = []
        permutation_all = []
        permutation_median = []
        permutation_lower_conf_limit = []
        permutation_upper_conf_limit = []
        for col, importances in permutation_feature_importances.items():
            cols.append(col)
            permutation_median.append(np.median(importances))
            permutation_lower_conf_limit.append(np.percentile(importances, 2.5))
            permutation_upper_conf_limit.append(np.percentile(importances, 97.5))
            for importance in importances:
                cols_all.append(col)
                permutation_all.append(importance)

        plt.clf()
        plt.figure(figsize=(15,6))
        sns.barplot(x=cols, y=permutation_median)
        sns.stripplot(x=cols, y=permutation_lower_conf_limit, edgecolor='k',
                      linewidth=1, s=6, jitter=False)
        sns.stripplot(x=cols, y=permutation_upper_conf_limit, edgecolor='k',
                      linewidth=1, s=6, jitter=False)
        plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
        plt.xlabel('Barrel')
        plt.ylabel('Importance')
        plt.savefig('{}/{}Permutation_feat_importance_percentiles.svg'.format(
            self.results_dir, plt_name
        ))
        plt.show()

        plt.clf()
        plt.figure(figsize=(15,6))
        sns.stripplot(x=cols_all, y=permutation_all, edgecolor='k', linewidth=1,
                      size=2.5, alpha=0.2)
        plt.xticks(np.arange(len(cols)), cols, rotation='vertical')
        plt.xlabel('Barrel')
        plt.ylabel('Importance')
        plt.savefig('{}/{}Permutation_feat_importance_all_data.svg'.format(
            self.results_dir, plt_name
        ))
        plt.show()

    def run_pca(self, x, scale):
        """
        Runs Principal Component Analysis and makes scatter plot of number of
        components vs. amount of information captured

        Input
        --------
        - x: Numpy array of x values

        Output
        --------
        - model: PCA model fitted to x values
        """

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import RobustScaler

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
        plt.savefig('{}/PCA_scree_plot.svg'.format(self.results_dir))
        plt.show()

        return model

    def define_fixed_model_params(self, clf):
        """
        For the 6 default ML algorithms run by this code (LogisticRegression,
        KNeighborsClassifier, GaussianNB, LinearSVC, SVC and
        RandomForestClassifier), defines a dictionary of hyperparameters and
        their corresponding values that will remain fixed throughout
        optimisation and training

        Input
        --------
        - clf: The selected ML algorithm,
        e.g. sklearn.ensemble.RandomForestClassifier()

        Output
        --------
        - params: Dictionary of fixed hyperparameters
        """

        if type(clf).__name__ == 'LogisticRegression':
            params = OrderedDict({'n_jobs': -1})
        elif type(clf).__name__ == 'KNeighborsClassifier':
            params = OrderedDict({'metric': 'minkowski',
                                  'n_jobs': -1})
        elif type(clf).__name__ == 'LinearSVC':
            params = OrderedDict({'dual': False})  # Change back to True
            # (= default) if n_samples < n_features
        elif type(clf).__name__ == 'SVC':
            params = OrderedDict()
        elif type(clf).__name__ == 'RandomForestClassifier':
            params = OrderedDict({'n_jobs': -1})
        elif type(clf).__name__ == 'GaussianNB':
            params = OrderedDict()

        return params

    def define_tuned_model_params(self, clf, x_train, n_folds):
        """
        For the 6 default ML algorithms run by this code (LogisticRegression,
        KNeighborsClassifier, GaussianNB, LinearSVC, SVC and
        RandomForestClassifier), returns dictionary of a sensible range of
        values for variable hyperparameters to be tested in randomised / grid
        search

        Input
        --------
        - clf: The selected ML algorithm,
        e.g. sklearn.ensemble.RandomForestClassifier()
        - x_train: Numpy array of x values of training data
        - n_folds: The number of folds to be used in k-folds cross-validation

        Output
        --------
        - params: Dictionary of parameter ranges that can be fed into
        RandomizedSearchCV or GridSearchCV
        """

        shape = x_train.shape[0]
        if type(clf).__name__ == 'LogisticRegression':
            params = OrderedDict({
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'sag', 'saga', 'newton-cg', 'lbfgs'],
                'multi_class': ['ovr', 'multinomial'],
                'C': np.logspace(-3, 5, 17)
            })
        elif type(clf).__name__ == 'KNeighborsClassifier':
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
        elif type(clf).__name__ == 'LinearSVC':
            params = OrderedDict({'C': np.logspace(-5, 15, num=41, base=2)})
        elif type(clf).__name__ == 'SVC':
            # For speed reasons (some kernels take a prohibitively long time to
            # train) am sticking with the default kernel ('rbf')
            params = OrderedDict({
                'C': np.logspace(-5, 15, num=41, base=2),
                'gamma': np.logspace(-15, 3, num=37, base=2)
            })
        elif type(clf).__name__ == 'RandomForestClassifier':
            # FIX - CHANGE RANDOM FOREST CLASSIFIER TO ADABOOST CLASSIFIER, AS THE FORMER TAKES TOO LONG TO RUN
            if (1/n_folds)*shape < 2:
                raise AlgorithmError(
                    'Too few data points in dataset to use random forest '
                    'classifier'
                )
            else:
                n_estimators = [int(x) for x in np.logspace(1, 4, 7)]
                min_samples_split = np.array([
                    int(x) for x in
                    np.linspace(2, int((1/n_folds)*shape), int((1/n_folds)*shape) - 1)
                ])
                min_samples_leaf = np.array([
                    int(x) for x in
                    np.linspace(2, int((1/n_folds)*0.5*shape), int((1/n_folds)*0.5*shape) - 1)
                ])
                params = OrderedDict({
                    'n_estimators': n_estimators,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                })
        elif type(clf).__name__ == 'GaussianNB':
            params = OrderedDict()

        return params

    def flag_extreme_params(self, best_params, poss_params):
        """
        Flags a warning if hyperparameter optimisation selects a value at the
        extreme end of an input (numerical) range

        Input
        --------
        - best_params: Dictionary of 'optimal' parameters returned by
        hyperparameter optimisation algorithm (e.g. RandomizedSearchCV or
        GridSearchCV)
        - poss_params: Dictionary of hyperparameter ranges fed into the
        hyperparameter optimisation algorithm, such as that returned by
        define_model_params
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
                        print('Warning: Optimal value selected for {} is at '
                              'the extreme of the range tested'.format(param))
                        print('Range tested: {}'.format(poss_vals))
                        print('Value selected: {}'.format(best_val))

    def conv_resampling_method(self, resampling_method):
        """
        """

        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.combine import SMOTEENN, SMOTETomek

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
        resampling_method, n_components_pca, params, scoring_func, n_iter=''
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
        --------
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

        Output
        --------
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
        --------
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
        --------
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
        --------
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
        --------
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
        --------
        - x_test: Numpy array of x values of test data
        - y_test: Numpy array of y values of test data
        - clf: Model previously fitted to the training data
        - test_scoring_funcs: Dictionary of sklearn scoring functions and
        dictionaries of arguments to be fed into these functions.
        E.g. sklearn.metrics.f1_score: {'average': 'macro'}
        - draw_conf_mat: Boolean, dictates whether or not confusion matrices are
        plotted.

        Output
        --------
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
        --------
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
        --------
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
            if self.randomise is True:
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
        --------
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
        --------
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
            if self.randomise is True:
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
