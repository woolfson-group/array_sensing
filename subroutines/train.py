

# Functions to perform ML analysis on parsed BADASS data.

import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

if __name__ == 'subroutines.train':
    from subroutines.parse_array_data import DefData
else:
    from array_sensing.subroutines.parse_array_data import DefData


class RunML(DefData):

    def __init__(
        self, peptide_list, results_dir, fluor_data, classes=None
    ):
        """
        - peptide_list: List of barrel names
        - results_dir: Path (either absolute or relative) to directory where
        output files should be saved. This directory will be created by the
        program and so should not already exist.
        - fluor_data: DataFrame of (standardised) fluorescence readings (output
        from parse_array_data class)
        - classes: List defining groups into which the analyte labels in the
        input data can be organised. For example, for a dataset of small
        molecule data, the analyte labels in the input data might be ['Lysine',
        'Alanine', 'Oleic acid', 'Butanoic acid', 'Glucose'], whilst the classes
        would be ['Amino acid', 'Amino acid', 'Fatty acid', 'Fatty acid',
        'Sugar']
        """

        DefData.__init__(self, peptide_list, results_dir)
        self.fluor_data = fluor_data.reset_index(drop=True)

        self.classes = classes
        if self.classes is None:
            self.subclass_split = False
        else:
            self.subclass_split = True
            if isinstance(self.classes, np.ndarray):
                pass
            else:
                if isinstance(self.classes, list):
                    self.classes = np.array(self.classes)
                else:
                    raise TypeError('classes argument should be either a 1d '
                                    'numpy array or a list')

        self.x = copy.deepcopy(self.fluor_data.drop('Analyte', axis=1)).to_numpy()
        if self.subclass_split is True:
            self.y = copy.deepcopy(self.classes)
            self.groups = np.array(
                ['{}_{}'.format(self.classes[n], self.fluor_data['Analyte'][n])
                 for n in range(self.fluor_data.shape[0])]
            )
            copy.deepcopy(self.fluor_data['Analyte']).to_numpy()
        elif self.subclass_split is False:
            self.y = copy.deepcopy(self.fluor_data['Analyte']).to_numpy()
            self.groups = None

    def scramble_class_labels(self):
        """
        Randomly shuffles y (and, if defined, linked group) values relative to
        x values (which are left unshuffled)
        """

        index = [n for n in range(self.fluor_data.shape[0])]
        random.shuffle(index)

        self.fluor_data['Analyte'] = self.fluor_data['Analyte'].iloc[index].tolist()
        self.y = self.y[index]
        if self.subclass_split is True:
            self.groups = self.groups[index]

    def randomly_pick_subclasses(self, percent_test):
        """
        Randomly selects subclasses to be saved as test set.
        N.B. Requires equal class sizes

        Input
        --------
        - percent_test: The percentage of the total classes to include in each
        subset

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

        subclasses = list(self.groups)
        classes = list(self.y)
        set_subclasses = list(set(subclasses))
        set_classes = list(set(classes))

        # Checks for equal class sizes
        organised_classes = {}
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
                'Multiplying the class size by percent_test gives a non-integer'
                ' value.\nPlease change the value of percent_test to allow '
                'division of the classes into subsets of equal (integer!) size'
            )
        else:
            num_test = int(num_test)  # range() requires integer values

        test_classes = [
            sub_class_array[i:i+num_test,].flatten('C').tolist() for i in
            [int(n) for n in range(0, sub_class_array.shape[0], num_test)]
        ]

        return test_classes

    def split_train_test_data(
        self, randomise, percent_test, test_analytes
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

        num_data_points = self.fluor_data.shape[0]

        self.randomise = randomise
        if self.randomise is True:
            num_test_data_points = round(num_data_points*percent_test)

            test_set = random.sample(range(num_data_points), num_test_data_points)
            train_set = [num for num in range(num_data_points) if not num in test_set]

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
        self.features = self.train_data.drop('Analyte', axis=1).columns.tolist()

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

    def calc_feature_correlations(self, train_data):
        """
        Calculates pairwise Spearman's rank correlation coefficient values
        between all 2-way combinations of features, and plots a heatmap.

        Input
        --------
        - train_data: DataFrame of the training data, with features as column
        names

        Output
        --------
        - correlation matrix: DataFrame of Spearman's rank correlation
        coefficient values for all pairwise combinations of features
        """

        feature_corr_df = train_data.drop('Analyte', axis=1)  # Must be a
        # dataframe, not a numpy array
        correlation_matrix = feature_corr_df.corr(method='spearman')

        plt.clf()
        heatmap = sns.heatmap(
            data=correlation_matrix, cmap='RdBu_r', annot=True,
            xticklabels=True, yticklabels=True
        )
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        plt.savefig('{}/Feature_correlations_Spearman_rank.svg'.format(self.results_dir))
        plt.show()

        return correlation_matrix

    def calc_feature_importances_kbest(
        self, x_train, y_train, features, method_classif
    ):
        """
        Runs a univariate statistical test (either f_classif or
        mutual_info_classif) between x and y to calculate the importances of the
        different input features to these scores, and plots these values.

        Input
        --------
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - features: List of feature (barrel) names
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

        model = SelectKBest(score_func=method_classif, k='all')
        model.fit(x_train, y_train)
        total_y = np.sum(model.scores_)
        norm_y = model.scores_ / total_y

        plt.clf()
        barplot = sns.barplot(x=features, y=norm_y)
        plt.xticks(
            np.arange(len(features)), features, rotation='vertical'
        )
        plt.savefig('{}/KBest_feature_importances_barplot.svg'.format(
            self.results_dir
        ))
        plt.show()

        plt.clf()
        order = np.argsort(-norm_y)
        lineplot = sns.lineplot(
            x=np.linspace(1, order.shape[0], order.shape[0]),
            y=np.cumsum(norm_y[order]), marker='o'
        )
        labels = np.concatenate((np.array(['']), np.array(features)[order]), axis=0)
        plt.xticks(
            np.arange(labels.shape[0]), labels, rotation='vertical'
        )
        ax = plt.gca()
        ax.set_ylim([0, 1.1])
        plt.savefig('{}/KBest_feature_importances_cumulative_plot.svg'.format(
            self.results_dir
        ))
        plt.show()

        score_df = pd.DataFrame({'Feature': features, 'Score': norm_y})
        score_df = score_df.sort_values(
            by=['Score'], axis=0, ascending=False
        ).reset_index(drop=True)

        return score_df

    def run_pca(self, x_train, features):
        """
        Runs Principal Component Analysis and makes scatter plot of number of
        components vs. amount of information captured

        Input
        --------
        - x_tain: Numpy array of x values of training data
        - features: List of feature (barrel) names
        """

        from sklearn.decomposition import PCA

        model = PCA()
        model.fit(x_train)
        max_num_components = len(features)

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

    def define_fixed_model_params(self, clf):
        """
        For the 6 default Ml algorithms run by this code (LogisticRegression,
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
            params = {'n_jobs': -1}
        elif type(clf).__name__ == 'KNeighborsClassifier':
            params = {'metric': 'minkowski',
                      'n_jobs': -1}
        elif type(clf).__name__ == 'LinearSVC':
            params = {'dual': False}  # Change back to True (= default) if
            # n_samples < n_features
        elif type(clf).__name__ == 'SVC':
            # For speed reasons (some kernels take a prohibitively long time to
            # train) am sticking with the default kernel ('rbf')
            params = {}
        elif type(clf).__name__ == 'RandomForestClassifier':
            params = {'n_jobs': -1}
        elif type(clf).__name__ == 'GaussianNB':
            params = {}

        return params

    def define_tuned_model_params(self, clf, x_train):
        """
        For the 6 default Ml algorithms run by this code (LogisticRegression,
        KNeighborsClassifier, GaussianNB, LinearSVC, SVC and
        RandomForestClassifier), returns dictionary of a sensible range of
        values for variable hyperparameters to be tested in randomised / grid
        search

        Input
        --------
        - clf: The selected ML algorithm,
        e.g. sklearn.ensemble.RandomForestClassifier()
        - x_train: Numpy array of x values of training data

        Output
        --------
        - params: Dictionary of parameter ranges that can be fed into
        RandomizedSearchCV or GridSearchCV
        """

        shape = x_train.shape[0]
        if type(clf).__name__ == 'LogisticRegression':
            params = {'penalty': ['l1', 'l2'],
                      'solver': ['liblinear', 'sag', 'saga', 'newton-cg', 'lbfgs'],
                      'multi_class': ['ovr', 'multinomial'],
                      'C': np.logspace(-3, 5, 17)}
        elif type(clf).__name__ == 'KNeighborsClassifier':
            neighbours = np.array(range(3, int(shape*0.2) + 1, 1))
            params = {'n_neighbors': neighbours,
                      'weights': ['uniform', 'distance'],
                      'p': np.array([1, 2])}
        elif type(clf).__name__ == 'LinearSVC':
            params = {'C': np.logspace(-5, 15, num=41, base=2)}
        elif type(clf).__name__ == 'SVC':
            # For speed reasons (some kernels take a prohibitively long time to
            # train) am sticking with the default kernel ('rbf')
            params = {'C': np.logspace(-5, 15, num=41, base=2),
                      'gamma': np.logspace(-15, 3, num=37, base=2)}
        elif type(clf).__name__ == 'RandomForestClassifier':
            n_estimators = [int(x) for x in np.logspace(1, 4, 7)]
            min_samples_split = np.array([
                int(x) for x in np.linspace(2, int(0.1*shape), int((0.1*shape) - 1))
            ])
            min_samples_leaf = np.array([
                int(x) for x in np.linspace(2, int(0.05*shape), int((0.05*shape) - 1))
            ])
            params = {'n_estimators': n_estimators,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
        elif type(clf).__name__ == 'GaussianNB':
            params = {}

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

    def run_randomised_search(
        self, x_train, y_train, train_groups, clf, splits, resampling_method,
        n_components_pca, params, scoring_func, n_iter=''
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
        including e.g. hyperparameters that are defined if other hyperparameters
        are defined as particular values. Returns the resuts of the randomised
        search.

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
        'max_sampling'; 'smote'; 'adasyn'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to tranform the data to after
        fitting the data with PCA
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

        # Determines number of iterations to run
        if n_iter == '':
            n_iter = 1
            for val in params.values():
                if isinstance(val, (list, np.ndarray)):
                    n_iter *= len(val)
            n_iter = int(n_iter*0.1)
            if n_iter < 25:
                n_iter = 25

        # Runs randomised search pipeline
        standardisation = StandardScaler()
        pca = PCA(n_components=n_components_pca)
        std_pca_clf = Pipeline([('std', standardisation),
                                ('PCA', pca),
                                ('resampling', resampling_method),
                                (type(clf).__name__, clf)])

        params = {'{}__{}'.format(type(clf).__name__, key): val
                  for key, val in params.items()}

        random_search = RandomizedSearchCV(
            estimator=std_pca_clf, param_distributions=params, n_iter=n_iter,
            scoring=scoring_func, n_jobs=-1, cv=splits, error_score=np.nan
        )
        random_search.fit(X=x_train, y=y_train, groups=train_groups)

        print('Randomised search with cross-validation results:')
        print('Best parameters: {}'.format(random_search.best_params_))
        print('Best score: {}'.format(random_search.best_score_))

        self.flag_extreme_params(random_search.best_params_, params)

        return random_search

    def run_grid_search(
        self, x_train, y_train, train_groups, clf, splits, resampling_method,
        n_components_pca, params, scoring_func
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
        including e.g. hyperparameters that are defined if other hyperparameters
        are defined as particular values. Returns the resuts of the grid search.

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
        'max_sampling'; 'smote'; 'adasyn'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to tranform the data to after
        fitting the data with PCA
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

        standardisation = StandardScaler()
        pca = PCA(n_components=n_components_pca)
        std_pca_clf = Pipeline([('std', standardisation),
                                ('PCA', pca),
                                ('resampling', resampling_method),
                                (type(clf).__name__, clf)])

        params = {'{}__{}'.format(type(clf).__name__, key): val
                  for key, val in params.items()}

        grid_search = GridSearchCV(
            estimator=std_pca_clf, param_grid=params, scoring=scoring_func,
            n_jobs=-1, cv=splits, error_score=np.nan
        )
        grid_search.fit(X=x_train, y=y_train, groups=train_groups)

        print('Grid search with cross-validation results:')
        print('Best parameters: {}'.format(grid_search.best_params_))
        print('Best score: {}'.format(grid_search.best_score_))

        self.flag_extreme_params(grid_search.best_params_, params)

        return grid_search

    def train_model(
        self, x_train, y_train, train_groups, clf, splits, resampling_method,
        n_components_pca, scoring_func
    ):
        """
        Trains user-specified model on the training data (without cross-
        validation). Also *separately* calculates how well the model fits the
        training data via cross-validation. Training data is first standardised
        (by subtracting the mean) and dividing by the standard deviation, then
        transformed to a user-specified number of features with PCA, and finally
        resampled if necessary to balance the class sizes, before it is fed into
        the model for training.

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
        'max_sampling'; 'smote'; 'adasyn'; 'smoteenn'; and 'smotetomek'.
        - n_components_pca: The number of components to tranform the data to
        after fitting the data with PCA
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
        - train_scores: The value of the selected scoring function calculated
        from cross-validation of the model on the training data.
        """

        from imblearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.model_selection import cross_val_score

        standardisation = StandardScaler()
        pca = PCA(n_components=n_components_pca)
        std_pca_clf = Pipeline([('std', standardisation),
                                ('PCA', pca),
                                ('resampling', resampling_method),
                                (type(clf).__name__, clf)])

        # Fits single model on all training data
        std_pca_clf.fit(X=x_train, y=y_train)

        # Calculates cross-validation score
        train_scores = cross_val_score(
            estimator=std_pca_clf, X=x_train, y=y_train, groups=train_groups,
            scoring=scoring_func, cv=splits, n_jobs=-1
        )
        print('Model cross-validation score: {}'.format(train_scores))

        return std_pca_clf, train_scores

    def test_model(
        self, x_test, y_test, clf, test_scoring_funcs
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
        test_scores = {}
        for func, params in test_scoring_funcs.items():
            if func.__name__ == 'cohen_kappa_score':
                params['y1'] = y_test
                params['y2'] = predictions
            else:
                params['y_true'] = y_test
                params['y_pred'] = predictions
            test_score = func(**params)
            test_scores[func.__name__] = test_score
            print('{}: {}'.format(func.__name__, test_score))

        # Below ensures that predicted and true labels are on the correct axes,
        # so think carefully before updating!
        plt.clf()
        labels = unique_labels(y_test, predictions)
        sns.heatmap(
            data=confusion_matrix(y_true=y_test, y_pred=predictions, labels=labels),
            cmap='RdBu_r', annot=True, xticklabels=True, yticklabels=True
        )
        ax = plt.gca()
        ax.set(xticklabels=labels, yticklabels=labels, xlabel='Predicted label',
               ylabel='True label')
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        plt.savefig('{}/{}_confusion_matrix.svg'.format(self.results_dir, type(clf).__name__))
        plt.show()

        return predictions, test_scores

    def run_algorithm(
        self, clf, x_train, y_train, train_groups, x_test, y_test,
        n_components_pca, run, params, train_scoring_func,
        test_scoring_funcs=None, resampling_method=['no_balancing'], n_iter='',
        cv_folds=5
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
        - n_components_pca: The number of components to tranform the data to
        after fitting the data with PCA
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
        'max_sampling'; 'smote'; 'adasyn'; 'smoteenn'; and 'smotetomek'.
        - n_iter: Integer number of hyperparameter combinations to test / ''. If
        set to '' will be set to either 25 or 10% of the total number of
        possible combinations of hyperparameter values specified in the params
        dictionary (whichever is larger).
        - cv_folds: Integer number of folds to run in cross-validation. E.g. as
        a default cv_folds = 5, which generates 5 folds of training and
        validation data, with 80% and 20% of the data forming the training and
        validation sets respectively.

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

        from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
        from imblearn.combine import SMOTEENN, SMOTETomek
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import GroupKFold

        searches = {}
        train_scores = {}
        test_scores = {}
        predictions = np.array([])

        for method in resampling_method:
            if method == 'no_balancing':
                resampling_obj = None
            elif method == 'max_sampling':
                resampling_obj = RandomOverSampler(sampling_strategy='not majority')
            elif method == 'smote':
                resampling_obj = SMOTE(sampling_strategy='not majority')
            elif method == 'adasyn':
                resampling_obj = ADASYN(sampling_strategy='not majority')
            elif method == 'smoteenn':
                resampling_obj = SMOTEENN(sampling_strategy='not majority')
            elif method == 'smotetomek':
                resampling_obj = SMOTETomek(sampling_strategy='not majority')
            else:
                raise ValueError('Resampling method {} not recognised'.format(method))

            if self.randomise is True:
                # There must be more than cv_folds instances of each dataset
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True)
                splits = skf.split(X=x_train, y=y_train)

            elif self.randomise is False:
                gkf = GroupKFold(n_splits=cv_folds)
                splits = skf.split(X=x_train, y=y_train, groups=train_groups)

            if run == 'randomsearch':
                search = self.run_randomised_search(
                    x_train, y_train, train_groups, clf, splits, resampling_obj,
                    n_components_pca, params, train_scoring_func, n_iter
                )
                searches[method] = search
            elif run == 'gridsearch':
                search = self.run_grid_search(
                    x_train, y_train, train_groups, clf, splits, resampling_obj,
                    n_components_pca, params, train_scoring_func
                )
                searches[method] = search
            elif run == 'train':
                search, train_score_subset = self.train_model(
                    x_train, y_train, train_groups, clf, splits, resampling_obj,
                    n_components_pca, train_scoring_func
                )
                predictions, test_score_subset = self.test_model(
                    x_test, y_test, search, test_scoring_funcs
                )
                searches[method] = search
                train_scores[method] = train_score_subset
                test_scores[method] = test_score_subset

        return searches, train_scores, test_scores, predictions

    def run_ml(
        self, clf, x_train, y_train, train_groups, x_test, y_test,
        n_components_pca, run, fixed_params, tuned_params, train_scoring_func,
        test_scoring_funcs=None, resampling_method=['no_balancing'], n_iter='',
        cv_folds=5
    ):
        """
        Fits an input sklearn classifier to the data.

        Input
        --------
        - clf: Selected classifier from the sklearn package,
        e.g. sklearn.ensemble.RandomForestClassifier
        - x_train: Numpy array of x values of training data
        - y_train: Numpy array of y values of training data
        - train_groups: Numpy array of group names of training data
        - x_test: Numpy array of x values of test data
        - y_test: Numpy array of y values of test data
        - n_components_pca: The number of components to tranform the data to
        after fitting the data with PCA
        - run: Either 'randomsearch', 'gridsearch' or 'train'. Directs the
        function whether to run cross-validation with RandomizedSearchCV or
        GridSearchCV to select a suitable combination of hyperparameter values,
        or whether to train and test the model.
        - fixed_params: Dictionary of hyperparameters and their selected values
        that remain constant regardless of the value of 'run' (e.g. n_jobs = -1)
        - tuned_params: If run == 'randomsearch' or 'gridsearch', tuned_params
        is a dictionary of hyperparameter values to search. Key = hyperparameter
        name (must match the name of the parameter in the selected clf class);
        Value = range of values to test for that hyperparameter - note that all
        numerical ranges must be supplied as numpy arrays in order to avoid
        throwing an error with the imblearn Pipeline() class. Else if
        run == 'train', params is also a dictionary of hyperparameters, but in
        this case a single value is provided for each hyperparameter as opposed
        to a range.
        - train_scoring_func: The function used to score the fitted classifier
        on the data set aside for validation during cross-validation. Either a
        function, or the name of one of the scoring functions in sklearn (see
        list of recognised names at https://scikit-learn.org/stable/modules/
        model_evaluation.html#scoring-parameter).
        - test_scoring_funcs: Required if run == 'train'. Dictionary of sklearn
        scoring functions and dictionaries of arguments to be fed into these
        functions. E.g. sklearn.metrics.f1_score: {'average': 'macro'}
        - resampling_method: Name of the method used to resample the data in an
        imbalanced dataset. Recognised method names are: 'no_balancing';
        'max_sampling'; 'smote'; 'adasyn'; 'smoteenn'; and 'smotetomek'.
        - n_iter: Required if run == 'randomsearch'. Integer number of
        hyperparameter combinations to test / ''. If set to '' will be set to
        either 25 or 10% of the total number of possible combinations of
        hyperparameter values specified in the params dictionary (whichever is
        larger).
        - cv_folds: Integer number of folds to run in cross-validation. E.g. as
        a default cv_folds = 5, which generates 5 folds of training and
        validation data, with 80% and 20% of the data forming the training and
        validation sets respectively.

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

        run = run.lower().replace(' ', '')
        if run in ['randomsearch', 'gridsearch']:
            clf = clf(**fixed_params)
            search, *_ = self.run_algorithm(
                clf, x_train, y_train, train_groups, x_test, y_test,
                n_components_pca, run, tuned_params, train_scoring_func,
                test_scoring_funcs, resampling_method, n_iter, cv_folds
            )
            return search

        elif run == 'train':
            params = fixed_params + tuned_params
            clf = clf(**params)
            search, train_scores, test_scores, predictions = self.run_algorithm(
                clf, x_train, y_train, train_groups, x_test, y_test,
                n_components_pca, run, {}, train_scoring_func,
                test_scoring_funcs, resampling_method, n_iter, cv_folds
            )
            return search, train_scores, test_scores, predictions
