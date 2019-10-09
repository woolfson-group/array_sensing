

# Functions to perform ML analysis on parsed BADASS data.

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

if __name__ == 'array_sensing.train':
    from array_sensing.parse_array_data import def_data
else:
    from sensing_array_paper.array_sensing.parse_array_data import def_data


class run_ml(def_data):

    def __init__(self, dir_path, repeat_names, peptide_list, data):
        """
        - dir_path: Path (either absolute or relative) to directory containing
        xlsx files of fluorescence readings
        - repeat_names: Names used to label different repeat readings of the
        same analytes in the xlsx file names. **NOTE: the same repeat name
        should be used for all analytes in the same repeat (hence date is
        probably not the best label unless all analytes in the repeat run were
        measured on the same day)**.
        - peptide_list: List of barrel names
        - data: Dataframe of (standardised) fluorescence readings (output from
        parse_array_data class)
        """
        def_data.__init__(self, dir_path, repeat_names, peptide_list)
        self.data = data

    def split_train_test_data(self, randomise, percent_test, test_analytes):
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
        names of the data points to be set aside for model testing
        """

        import random

        num_data_points = self.data.shape[0]

        if randomise is True:
            num_test_data_points = round(num_data_points*percent_test)

            test_set = random.sample(range(num_data_points), num_test_data_points)
            train_set = [num for num in range(num_data_points) if not num in test_set]

        elif randomise is False:
            test_set = []
            train_set = []

            for n in range(num_data_points):
                if self.data['Analyte'][n] in test_analytes:
                    test_set.append(n)
                else:
                    train_set.append(n)

        test_data = self.data.iloc[test_set].reset_index(drop=True)
        train_data = self.data.iloc[train_set].reset_index(drop=True)

        self.train_data = train_data
        self.test_data = test_data
        self.features = self.train_data.drop('Analyte', axis=1).columns.tolist()

        if randomise is True:
            from sklearn.model_selection import StratifiedKFold
            self.cv = StratifiedKFold(n_splits=len(self.repeats), shuffle=False)
        elif randomise is False:
            from sklearn.model_selection import GroupKFold
            self.cv = GroupKFold(n_splits=len(self.repeats))

        self.train_x = self.train_data.drop('Analyte', axis=1).to_numpy()
        self.test_x = self.test_data.drop('Analyte', axis=1).to_numpy()

        if randomise is True:
            self.train_y = self.train_data['Analyte'].to_numpy()
            self.train_group = None
            self.test_y = self.test_data['Analyte'].to_numpy()
            self.test_group = None
        elif randomise is False:
            self.train_y = np.array([val.split()[0] for val in self.train_data['Analyte'].tolist()])
            self.train_group = self.train_data['Analyte'].to_numpy()
            self.test_y = np.array([val.split()[0] for val in self.test_data['Analyte'].tolist()])
            self.test_group = self.test_data['Analyte'].to_numpy()

    def calc_feature_correlations(self, train_data):
        """
        """

        feature_corr_df = train_data.drop('Analyte', axis=1)  # Must be a
        # dataframe, not a numpy array
        correlation_matrix = feature_corr_df.corr(method='spearman')

        plt.clf()
        heatmap = sns.heatmap(
            data=correlation_matrix, cmap='RdBu_r', annot=True
        )
        plt.savefig('Feature_correlations_Spearman_rank.svg')
        plt.show()

        return correlation_matrix

    def calc_feature_importances_kbest(self, x_train, y_train, features, method):
        """
        method: either f_classif or mutual_info_classif
        """

        from sklearn.feature_selection import SelectKBest

        model = SelectKBest(score_func=method, k='all')
        model.fit(x_train, y_train)

        plt.clf()
        barplot = sns.barplot(x=features, y=model.scores_)
        plt.savefig('KBest_feature_importances_barplot.svg')
        plt.show()

        plt.clf()
        order = np.argsort(-model.scores_)
        lineplot = sns.lineplot(
            x=np.linspace(1, order.shape[0], order.shape[0]),
            y=np.cumsum(model.scores_[order]), marker='o'
        )
        labels = np.array(features)[order]
        ax = plt.gca()
        ax.set(xticks=labels)
        plt.savefig('KBest_feature_importances_cumulative_plot.svg')
        plt.show()

        score_df = pd.DataFrame({'Feature': features, 'Score': model.scores_})
        score_df = score_df.sort_values(by=['Score'], axis=0, ascending=False).reset_index(drop=True)

        return score_df

    def select_features(self, feat_retain):
        """
        features: list of features to retain
        """

        feat_remove = [pep for pep in self.features if not pep in feat_retain]
        self.sub_train_data = self.train_data.drop(feat_remove, axis=1)
        self.sub_test_data = self.test_data.drop(feat_remove, axis=1)
        self.sub_train_x = self.train_data.drop(feat_remove + ['Analyte'], axis=1).to_numpy()
        self.sub_test_x = self.test_data.drop(feat_remove + ['Analyte'], axis=1).to_numpy()
        self.sub_features = self.train_data.drop('Analyte', axis=1).columns.tolist()

    def run_pca(self, x_train, features):
        """
        Calculates principal components
        """

        from sklearn.decomposition import PCA

        model = PCA()
        model.fit(x_train)
        max_num_components = len(features)

        plt.clf()
        x_labels = np.linspace(1, max_num_components, max_num_components)
        lineplot = sns.lineplot(
            x_labels,
            np.cumsum(model.explained_variance_ratio_),
            marker='o'
        )
        ax = plt.gca()
        ax.set(xticks=x_labels)
        plt.savefig('PCA_scree_plot.svg')
        plt.show()

    def run_pca_and_transform(self, x_train, x_test, n_components):
        """
        """

        from sklearn.decomposition import PCA

        model = PCA(n_components=n_components)
        model.fit(x_train)
        self.pca_train_x = model.transform(x_train)
        self.pca_test_x = model.transform(x_test)
        self.pca_features = [num for num in range(1, n_components+1)]

    def recursive_feature_elimination_with_cross_validation(
        self, x_train, y_train, groups, x_test, model
    ):
        """
        NOTE: Number of features selected is heavily depedent upon the number of splits...
        model = a supervised learning estimator with a fit method that provides
        information about feature importances, via either a coef_ or a
        feature_importances_ attribute (examples incl. random forest, SVC with
        linear kernel, logistic regression, etc.)
        """

        from sklearn.feature_selection import RFECV

        # Use StratifiedKFold with shuffle=False (effectively group is now
        # date, so to prevent overfitting cross-validation shouldn't randomly
        # select data across different dates for training and validation, and
        # then test on data all collected on a different date)
        rfecv = RFECV(
            estimator=model, step=1, cv=self.cv, scoring='accuracy', n_jobs=-1
        )
        rfecv.fit(X=x_train, y=y_train, groups=groups)

        print('Optimal number of features : {}'.format(rfecv.n_features_))

        plt.clf()
        x_labels = np.linspace(1, len(rfecv.grid_scores_), len(rfecv.grid_scores_))
        sns.lineplot(x_labels, rfecv.grid_scores_, markers='o')
        ax = plt.gca()
        ax.set(xticks=x_labels)
        plt.xlabel('Number of features')
        plt.ylabel('Cross validation score')
        plt.savefig('{}_recursive_feature_elimination_cross_validation_scores.svg'.format(type(model).__name__))
        plt.show()

        self.rfecv_train_x = rfecv.transform(x_train)
        self.rfecv_test_x = rfecv.transform(x_test)
        self.rfecv_features = [
            self.features[i[0]] for i, n in np.ndenumerate(rfecv.ranking_) if n == 1
        ]

    def define_model_params(self, clf, x_train):
        """
        Sets parameters to try with RandomizedSearchCV
        """

        shape = x_train.shape[0]
        if type(clf).__name__ == 'LogisticRegression':
            params = {'penalty': ['l1', 'l2'],
                      'solver': ['liblinear', 'sag', 'saga', 'newton-cg', 'lbfgs'],
                      'multi_class': ['ovr', 'multinomial'],
                      'C': np.logspace(-3, 5, 17)}
        elif type(clf).__name__ == 'KNeighborsClassifier':
            neighbours = list(range(3, int(shape*0.2) + 1, 1))
            params = {'n_neighbors': neighbours,
                      'weights': ['uniform', 'distance'],
                      'p': [1, 2]}
        elif type(clf).__name__ == 'SVC':
            params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'degree': np.linspace(2, 5, 4),
                      'C': np.logspace(-3, 5, 17),
                      'gamma': np.logspace(-3, 5, 17)}
        elif type(clf).__name__ == 'RandomForestClassifier':
            n_estimators = [int(x) for x in np.logspace(1, 4, 7)]
            min_samples_split = [
                int(x) for x in np.linspace(2, int(0.1*shape), int((0.1*shape) - 1))
            ]
            min_samples_leaf = [
                int(x) for x in np.linspace(2, int(0.05*shape), int((0.05*shape) - 1))
            ]
            params = {'n_estimators': n_estimators,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}

        return params

    def flag_extreme_params(self, best_params, poss_params):
        """
        """

        for param, best_val in best_params.items():
            poss_vals = poss_params[param]
            if all(isinstance(poss_val, (int, float)) for poss_val in poss_vals) and len(poss_vals) > 2:
                if best_val in [poss_vals[0], poss_vals[-1]]:
                    print('Warning: Optimal value selected for {} is at the extreme of the range tested'.format(param))
                    print('Range tested: {}'.format(poss_vals))
                    print('Value selected: {}'.format(best_val))

    def run_randomized_search(self, x_train, y_train, train_groups, clf, params):
        """
        """

        from sklearn.model_selection import RandomizedSearchCV

        n_iter = 1
        for val in list(params.values()):
            n_iter *= len(val)
        n_iter = int(n_iter*0.2)
        if n_iter < 25:
            n_iter = 25

        random_search = RandomizedSearchCV(
            estimator=clf, param_distributions=params, n_iter=n_iter,
            scoring='accuracy', n_jobs=-1, cv=self.cv, iid=False,
            error_score=np.nan
        )
        random_search.fit(X=x_train, y=y_train, groups=train_groups)

        print('Randomised search with cross-validation results:')
        print('Best parameters: {}'.format(random_search.best_params_))
        print('Best score: {}'.format(random_search.best_score_))

        self.flag_extreme_params(random_search.best_params_, params)

        return random_search

    def run_grid_search(self, x_train, y_train, train_groups, clf, params):
        """
        """

        from sklearn.model_selection import GridSearchCV

        grid_search = GridSearchCV(
            estimator=clf, param_grid=params, scoring='accuracy', n_jobs=-1,
            cv=self.cv, iid=False, error_score=np.nan
        )
        grid_search.fit(X=x_train, y=y_train, groups=train_groups)
        print('Grid search with cross-validation results:')
        print('Best parameters: {}'.format(grid_search.best_params_))
        print('Best score: {}'.format(grid_search.best_score_))

        self.flag_extreme_params(grid_search.best_params_, params)

        return grid_search

    def train_model(self, x_train, y_train, train_groups, clf):
        """
        """

        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(
            estimator=clf, X=x_train, y=y_train, groups=train_groups,
            scoring='accuracy', cv=self.cv, n_jobs=-1, error_score=np.nan
        )
        print('Model cross-validation score: {}'.format(clf.score))

        clf.fit(x_train, y_train)

        return clf

    def test_model(self, x_test, y_test, clf):
        """
        """

        from sklearn.metrics import (
            accuracy_score, recall_score, precision_score, confusion_matrix
        )

        predictions = clf.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
        recall = recall_score(y_true=y_test, y_pred=predictions, average='micro')
        precision = precision_score(y_true=y_test, y_pred=predictions, average='micro')

        print('Accuracy: {}'.format(accuracy))
        print('Recall: {}'.format(recall))
        print('Precision: {}'.format(precision))

        plt.clf()
        labels=list(set(y_test))
        sns.heatmap(
            data=confusion_matrix(y_true=y_test, y_pred=predictions, labels=labels),
            cmap='RdBu_r', annot=True
        )
        ax = plt.gca()
        ax.set(xticklabels=labels, yticklabels=labels, xlabel='Predicted label',
               ylabel='True label')
        plt.savefig('{}_confusion_matrix.svg'.format(type(clf).__name__))
        plt.show()

        return accuracy, recall, precision

    def run_algorithm(self, clf, x_train, y_train, train_groups, x_test, y_test, run, params):
        """
        """

        if run == 'randomsearch':
            search = self.run_randomized_search(x_train, y_train, train_groups, clf, params)
            return search
        elif run == 'gridsearch':
            search = self.run_grid_search(x_train, y_train, train_groups, clf, params)
            return search
        elif run == 'train':
            search = self.train_model(x_train, y_train, train_groups, clf)
            accuracy, recall, precision = self.test_model(x_test, y_test, search)
            return search, accuracy, recall, precision

    def loop_run_algorithm(self):
        """
        To be run for training and testing - but need manual input for training...
        So perhaps use this function to generate the analyte combinations, but then run them manually.
        Need equal numbers of varieties (sub_group) in each group, e.g. need 10
        types of green, black and grey tea
        """

        shuffled_data = []
        groups = set(train_groups)
        sub_groups = set(y_train)
        for group in groups:
            shuffled_data += [
                sub_group for sub_group in sub_groups if sub_group.startswith(group)
            ]
        shuffled_data = np.array(shuffled_data)

        for n in range(shuffled_data.shape[0]):
            np.random.shuffle(shuffled_data[n])

        analyte_list = []
        for n in range(shuffled_data.shape[1]):
            analytes = shuffled_data.transpose()[n]
            analyte_list.append(analytes)

        return analyte_list

    def run_logistic_regression(
        self, x_train, y_train, train_groups, x_test, y_test, run, params
    ):
        """
        """

        from sklearn.linear_model import LogisticRegression

        run = run.lower().replace(' ', '')
        if run in ['randomsearch', 'gridsearch']:
            clf = LogisticRegression(n_jobs=-1)
        elif run == 'train':
            params['n_jobs'] = -1
            clf = LogisticRegression(**params)

        search = self.run_algorithm(
            clf, x_train, y_train, train_groups, x_test, y_test, run, params
        )

        return search


    def run_k_nearest_neighbours(
        self, x_train, y_train, train_groups, x_test, y_test, run, params
    ):
        """
        """

        from sklearn.neighbors import KNeighborsClassifier

        run = run.lower().replace(' ', '')
        if run in ['randomsearch', 'gridsearch']:
            clf = KNeighborsClassifier(metric='minkowski', n_jobs=-1)
        elif run == 'train':
            params['metric'] = 'minkowski'
            params['n_jobs'] = -1
            clf = KNeighborsClassifier(**params)

        search = self.run_algorithm(
            clf, x_train, y_train, train_groups, x_test, y_test, run, params
        )

        return search


    def run_naive_Bayes(self, x_train, y_train, train_groups, x_test, y_test):
        """
        """

        from sklearn.naive_bayes import GaussianNB

        # No hyperparameters to optimise
        clf = GaussianNB()
        search = self.run_algorithm(
            clf, x_train, y_train, train_groups, x_test, y_test, run='train',
            params=None
        )

        return search

    def run_svc(
        self, x_train, y_train, train_groups, x_test, y_test, run, params
    ):
        """
        Parameters = c, gamma, kernel, degree
        """

        from sklearn.svm import SVC

        run = run.lower().replace(' ', '')
        if run in ['randomsearch', 'gridsearch']:
            clf = SVC()
        elif run == 'train':
            clf = SVC(**params)

        search = self.run_algorithm(
            clf, x_train, y_train, train_groups, x_test, y_test, run, params
        )

        return search

    def run_random_forest(
        self, x_train, y_train, train_groups, x_test, y_test, run, params
    ):
        """
        """

        from sklearn.ensemble import RandomForestClassifier

        run = run.lower().replace(' ', '')
        if run in ['randomsearch', 'gridsearch']:
            clf = RandomForestClassifier(n_jobs=-1)
        elif run == 'train':
            params['n_jobs'] = -1
            clf = RandomForestClassifier(**params)

        search = self.run_algorithm(
            clf, x_train, y_train, train_groups, x_test, y_test, run, params
        )

        return search

    """
    Possible metrics include accuracy (= (TP + TN) / (TP + TN + FN + FP), good for when cost of false positive and false negative are equally high) (can use balanced accuracy score if the classes are not of equal size / weight), Matthews correlation coefficient (can
    be used even if the classes are of v. different sizes, but this is not
    the case here), F1 score (The F1 score can be interpreted as a weighted
    average of the precision and recall, where an F1 score reaches its best
    value at 1 and worst score at 0. The relative contribution of precision
    and recall to the F1 score are equal.
    F1 = ((2*precision*recall) / (precision + recall), good for when FP and FN cost is equally high plus class distribution is uneven), precision (= TP / (TP + FP), good for when cost of false positive is high), recall (= TP / (TP + FN), good for when cost of false negative is high),
    Hamming loss (the fraction of labels incorrectly predicted = (FP + FN) / (FP + FN + TP + TN)) (= 1 - accuracy)
    Don't use Jaccard score since makes sets of predicted + actual labels => order is lost
    Zero one loss score splits data into subsets, gives pair a 1 if pair is exact match, otherwise 0. Returns score equal to the fraction of imperfectly predicted subsets. Don't think I want to use this metric.
    Should consider hinge_loss for SVM (considers only prediction errors, used to score models generated from maximal margin algorithms)
    Should consider log_loss for logistic regression


    accuracy = (TP + TN) / (TP + TN + FP + FN) (i.e. fraction correct)
    sensitivity = TP / (TP + FN) (i.e. fraction positive correctly predicted)
    specificity = TN / (TN + FP) (i.e. fraction negative correctly predicted)
    precision = TP / (TP + FP) (i.e. fraction predicted positive that are positive)
    recall = TP / (TP + FN) (i.e. same as sensitivity)
    roc (receiver operating curve) = true positive rate (TP / (TP + FN)) (y-axis) vs false positive rate (FP / (FP + TN)) (x-axis), roc plot shows trade-off between sensitivity and specificity
    """