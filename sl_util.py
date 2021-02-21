import copy
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import config_util


def load_data(file_name):
    csv_path = os.path.join("data", file_name)
    return pd.read_csv(csv_path)


def get_chart_path(file_name):
    return os.path.join("charts", file_name)


def get_dataset_x(dataset):
    return dataset.iloc[:, :-1]


def get_dataset_y(dataset):
    return dataset.iloc[:, -1:]


def get_train_set_x(dataset_dict):
    return dataset_dict["train_set_x"]


def get_train_set_y(dataset_dict):
    return dataset_dict["train_set_y"]


def get_test_set_x(dataset_dict):
    return dataset_dict["test_set_x"]


def get_test_set_y(dataset_dict):
    return dataset_dict["test_set_y"]


def to_series(df):
    return df.iloc[:, -1]


def split_train_test(dataset):
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=config_util.ran_state)
    train_set_x = get_dataset_x(train_set)
    train_set_y = get_dataset_y(train_set)
    test_set_x = get_dataset_x(test_set)
    test_set_y = get_dataset_y(test_set)
    return {"train_set_x": train_set_x,
            "train_set_y": train_set_y,
            "test_set_x": test_set_x,
            "test_set_y": test_set_y}


def get_column_names(dataset):
    return dataset.columns.values.tolist()


def get_unique_values(dataseries):
    return dataseries.unique()


def convert_ints_to_strs(ints):
    return [str(int_val) for int_val in ints]


def scale_attributes(data_dict):
    scaler = StandardScaler()
    data_dict["train_set_x"][:] = scaler.fit_transform(data_dict["train_set_x"].values)
    data_dict["test_set_x"][:] = scaler.transform(data_dict["test_set_x"].values)


def get_output_distribution(dataset):
    df = pd.concat([to_series(dataset).value_counts(),
                    to_series(dataset).value_counts(normalize=True).mul(100)],
                   axis=1, keys=('# of Instances', 'Percentage(%)'))
    df.index.name = 'Class'
    return df


def shuffle_df(dataset, times):
    for i in range(times):
        dataset = shuffle(dataset, random_state=config_util.ran_state)
    return dataset


def get_decision_tree_classifier(max_leaf_nodes=None, min_samples_leaf=1, max_depth=None, criterion="entropy"):
    return DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                  max_depth=max_depth,
                                  criterion=criterion,
                                  random_state=config_util.ran_state)


def get_neural_network_classifier(max_iter=100, warm_start=False, learning_rate_init=0.001, momentum=0.9,
                                  hidden_layer_sizes=(100,), solver='sgd', early_stopping=False):
    return MLPClassifier(solver=solver, max_iter=max_iter, warm_start=warm_start, learning_rate_init=learning_rate_init,
                         momentum=momentum, hidden_layer_sizes=hidden_layer_sizes, early_stopping=early_stopping,
                         random_state=config_util.ran_state)


def get_adaboost_classifier(n_estimators=50, learning_rate=1):
    # use ada-boost because it's easier
    return AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                              random_state=config_util.ran_state)


def get_svm_classifier(kernel="sigmoid", max_iter=-1, degree=3, C=1.0):
    return SVC(kernel=kernel, max_iter=max_iter, degree=degree, C=C, random_state=config_util.ran_state)


def get_knn_classifier(n_neighbors=10, leaf_size=30, p=2):
    return KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, p=p)


def plot_hist_classes(dataset, title, file_name):
    plt.clf()
    to_series(dataset).value_counts().plot(kind='bar', width=0.2, align='center')
    plt.xlabel('Class')
    plt.ylabel('# of Instances')
    plt.xticks(rotation='horizontal')
    plt.title(title)
    plt.savefig(get_chart_path(file_name))


def plot_correlation(dataset, title, file_name):
    corr_matrix = dataset.corr()
    plt.clf()
    plt.figure(figsize=(8, 8))
    heatmap = sns.heatmap(data=corr_matrix, vmin=corr_matrix.values.min(), vmax=1, annot=True, cmap="Blues",
                          cbar=False,
                          annot_kws={"fontsize": 12})
    heatmap.set_title(title, fontdict={'fontsize': 12})
    plt.xticks(rotation=50, fontsize=10)
    plt.yticks(rotation=70, fontsize=10)
    plt.tight_layout()
    plt.savefig(get_chart_path(file_name))


def graph_decision_tree(tree_clf, data_dict, title):
    x_set = get_train_set_x(data_dict)
    y_set = get_train_set_y(data_dict)
    tree_clf.fit(x_set, y_set)
    export_graphviz(tree_clf,
                    out_file=get_chart_path(title) + ".dot",
                    feature_names=get_column_names(x_set),
                    class_names=convert_ints_to_strs(
                        get_unique_values(to_series(y_set))),
                    rounded=True,
                    filled=True)


def plot_lc_mcc(x_values,
                train_scores_mean, train_scores_std,
                validation_scores_mean, validation_scores_std,
                dataset_index, model_index, plot_index,
                y_lim, params, version, xy_label, is_marked, is_fill_between):
    plt.clf()
    if params is not None:
        title = "%s of %s for %s\n%s" % (
            config_util.plot_names[plot_index], config_util.sl_model_names[model_index],
            config_util.dataset_names[dataset_index], params)
    else:
        title = "%s of %s for %s" % (
            config_util.plot_names[plot_index], config_util.sl_model_names[model_index],
            config_util.dataset_names[dataset_index])
    file_name = "%s_%s_%s_%s_%s.png" % (
        config_util.dataset_file_names[dataset_index], config_util.sl_model_file_names[model_index],
        config_util.plot_file_names[plot_index], xy_label[3],
        version)
    # plt.figure(figsize=[6, 5])

    if is_marked:
        plt.plot(x_values, train_scores_mean,
                 'o-', markersize=5,
                 color=config_util.plot_color[plot_index][0], label="Training %s" % xy_label[0])
        plt.plot(x_values, validation_scores_mean,
                 'o-', markersize=5,
                 color=config_util.plot_color[plot_index][1], label="Cross-validation %s" % xy_label[0])
    else:
        plt.plot(x_values, train_scores_mean, color=config_util.plot_color[plot_index][0],
                 label="Training %s" % xy_label[0])
        plt.plot(x_values, validation_scores_mean,
                 color=config_util.plot_color[plot_index][1], label="Cross-validation %s" % xy_label[0])

    if is_fill_between:
        plt.fill_between(x_values,
                         train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color=config_util.plot_color[plot_index][0])
        plt.fill_between(x_values,
                         validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std,
                         alpha=0.1, color=config_util.plot_color[plot_index][1])
    plt.ylabel(xy_label[0], fontsize=14)
    plt.xlabel(xy_label[1], fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    plt.ylim(y_lim[0], y_lim[1])
    plt.tight_layout()
    plt.savefig(get_chart_path(file_name))


def plot_tc(x_values,
            times_mean, times_std,
            dataset_index, model_index, plot_index,
            y_lim, params, version, xy_label, is_marked, is_fill_between):
    plt.clf()
    title = "%s of %s for %s" % (
        config_util.plot_names[plot_index], config_util.sl_model_names[model_index],
        config_util.dataset_names[dataset_index])
    file_name = "%s_%s_%s_%s_%s.png" % (
        config_util.dataset_file_names[dataset_index], config_util.sl_model_file_names[model_index],
        config_util.plot_file_names[plot_index], xy_label[3],
        version)
    # plt.figure(figsize=[6, 5])

    if is_marked:
        plt.plot(x_values, times_mean,
                 'o-', markersize=5,
                 color=config_util.plot_color[plot_index][0], label="Fit Time")
    else:
        plt.plot(x_values, times_mean, color=config_util.plot_color[plot_index][0], label="Fit Time")

    if is_fill_between:
        plt.fill_between(x_values,
                         times_mean - times_std, times_mean + times_std,
                         alpha=0.1, color=config_util.plot_color[plot_index][0])

    plt.ylabel(xy_label[0], fontsize=14)
    plt.xlabel(xy_label[1], fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    plt.ylim(y_lim[0], y_lim[1])
    plt.tight_layout()
    plt.savefig(get_chart_path(file_name))


def print_mean_scores(train_sizes, train_scores_mean, validation_scores_mean, dataset_index, model_index, cv_val):
    print("%s Model of %s (CV=%s)" % (
        config_util.sl_model_names[model_index], config_util.dataset_names[dataset_index], str(cv_val)))
    print('Mean training scores\n', pd.Series(train_scores_mean, index=train_sizes))
    print('Mean validation scores\n', pd.Series(validation_scores_mean, index=train_sizes))
    print('-' * 20)  # separator


def print_dataset_info(dataset, dataset_index):
    # number of instances, number and type of features
    print("Features of %s Dataset" % config_util.dataset_names[dataset_index])
    print(get_dataset_x(dataset).info())


def print_description(data_dict, dataset_index, pre_text):
    # mean and std of features
    clos = ['fixed_acidity', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'alcohol']
    print('%s Standardization: Description of %s\n' % (pre_text, config_util.dataset_names[dataset_index]))
    print(get_train_set_x(data_dict)[clos].describe())


def exe_grid_search(classifier, data_dict, param_grid):
    cv_val = 5
    x_train = get_train_set_x(data_dict)
    y_train = get_train_set_y(data_dict)
    grid_search = GridSearchCV(classifier, param_grid, cv=cv_val,
                               scoring='accuracy',
                               return_train_score=True)

    grid_search.fit(x_train, y_train)

    # output
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)


def generate_learning_curves_sizes_times(classifier, data_dict, dataset_index, model_index, y_lim, params, version,
                                         is_fill_between=True):
    cv_val = 5
    # create learning curve
    train_sizes, train_scores, validation_scores, fit_times, _ = learning_curve(
        estimator=classifier,
        X=get_train_set_x(data_dict),
        y=(get_train_set_y(data_dict)).values.ravel(),
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=cv_val,
        # shuffle=True,
        random_state=config_util.ran_state,
        scoring='accuracy',
        return_times=True)

    # calculate mean
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    fit_times_cum_mean = np.cumsum(fit_times_mean, axis=0)

    # calculate error rate
    train_error_rates = 1.0 - train_scores
    validation_error_rates = 1.0 - validation_scores
    train_error_rates_mean = np.mean(train_error_rates, axis=1)
    train_error_rates_std = np.std(train_error_rates, axis=1)
    validation_error_rates_mean = np.mean(validation_error_rates, axis=1)
    validation_error_rates_std = np.std(validation_error_rates, axis=1)

    # print mean
    # print_mean_scores(train_sizes, train_scores_mean,
    #                   validations_scores_mean, dataset_index,
    #                   model_index, cv_val)

    # plot Accuracy vs train_sizes
    plot_lc_mcc(train_sizes,
                train_scores_mean, train_scores_std,
                validation_scores_mean, validation_scores_std,
                dataset_index, model_index, 0,
                y_lim[0], params, version + "_a",
                ('Accuracy', 'Training Size', 'Accuracy vs Training Size', 'size'),
                is_marked=True,
                is_fill_between=is_fill_between
                )
    # plot error rate vs train_sizes
    plot_lc_mcc(train_sizes,
                train_error_rates_mean, train_error_rates_std,
                validation_error_rates_mean, validation_error_rates_std,
                dataset_index, model_index, 0,
                y_lim[1], params, version + "_e",
                ('Error Rate', 'Training Size', 'Error Rate vs Training Size', 'size'),
                is_marked=True,
                is_fill_between=is_fill_between
                )
    if model_index != 1:
        # plot Accuracy vs Fit times
        plot_lc_mcc(fit_times_cum_mean,
                    train_scores_mean, train_scores_std,
                    validation_scores_mean, validation_scores_std,
                    dataset_index, model_index, 2,
                    y_lim[0], params, version + "_a",
                    ('Accuracy', 'fit_time (seconds)', 'Accuracy vs fit_time', 'fittime'),
                    is_marked=True,
                    is_fill_between=is_fill_between
                    )

        # plot Fit Times vs train_sizes
        plot_tc(train_sizes,
                fit_times_cum_mean, fit_times_std,
                dataset_index, model_index, 1,
                y_lim[2], params, version,
                ('Fit Time (seconds)', 'Training Size', 'Fit Time vs Training Size', 'size'),
                is_marked=True,
                is_fill_between=is_fill_between
                )


def generate_learning_curves_iterations(classifier, max_iteration, data_dict, dataset_index, model_index, y_lim,
                                        params,
                                        version, is_fill_between=True):
    cv_val = 5
    max_iteration = max_iteration
    skfolds = StratifiedKFold(n_splits=cv_val)
    x_train = get_train_set_x(data_dict)
    y_train = get_train_set_y(data_dict)
    train_scores, validation_scores = [], []

    for train_index, validation_index in skfolds.split(x_train, y_train):
        clone_clf = copy.deepcopy(classifier)
        x_train_folds = x_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        x_validation_fold = x_train.iloc[validation_index]
        y_validation_fold = y_train.iloc[validation_index]

        train_scores_fold, validation_scores_fold = [], []
        for iteration in range(1, max_iteration + 1):
            clone_clf.fit(X=x_train_folds, y=y_train_folds.values.ravel())

            train_predict = clone_clf.predict(x_train_folds)
            validation_predict = clone_clf.predict(x_validation_fold)

            train_score = accuracy_score(y_train_folds, train_predict)
            validation_score = accuracy_score(y_validation_fold, validation_predict)

            train_scores_fold.append(train_score)
            validation_scores_fold.append(validation_score)

        train_scores.append(train_scores_fold)
        validation_scores.append(validation_scores_fold)

    train_scores_mean = np.mean(train_scores, axis=0)
    train_scores_std = np.std(train_scores, axis=0)
    validation_scores_mean = np.mean(validation_scores, axis=0)
    validation_scores_std = np.std(validation_scores, axis=0)

    # calculate error rate
    train_error_rates = 1.0 - np.array(train_scores)
    validation_error_rates = 1.0 - np.array(validation_scores)
    train_error_rates_mean = np.mean(train_error_rates, axis=0)
    train_error_rates_std = np.std(train_error_rates, axis=0)
    validation_error_rates_mean = np.mean(validation_error_rates, axis=0)
    validation_error_rates_std = np.std(validation_error_rates, axis=0)

    # plot accuracy vs iterations
    plot_lc_mcc(range(1, max_iteration + 1),
                train_scores_mean, train_scores_std,
                validation_scores_mean, validation_scores_std,
                dataset_index, model_index, 4,
                y_lim[0], params, version + "_a",
                ('Accuracy', 'Iteration', 'Accuracy vs Iteration', 'iteration'),
                is_marked=False,
                is_fill_between=is_fill_between
                )

    # plot error rate vs iterations - error rate
    plot_lc_mcc(range(1, max_iteration + 1),
                train_error_rates_mean, train_error_rates_std,
                validation_error_rates_mean, validation_error_rates_std,
                dataset_index, model_index, 5,
                y_lim[1], params, version + "_e",
                ('Loss', 'Iteration', 'Loss vs Iteration', 'iteration'),
                is_marked=False,
                is_fill_between=is_fill_between
                )


def generate_model_complexity_curves(classifier, param_name, param_range, data_dict,
                                     dataset_index, model_index, y_lim, params, version, is_fill_between=True,
                                     is_iteration=False, is_seq_in_range=False):
    cv_val = 5
    train_scores, validation_scores = validation_curve(
        estimator=classifier,
        X=get_train_set_x(data_dict),
        y=(get_train_set_y(data_dict)).values.ravel(),
        param_name=param_name,
        param_range=param_range,
        cv=cv_val,
        scoring='accuracy')

    # calculate mean
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    # calculate error rate
    train_error_rates = 1.0 - train_scores
    validation_error_rates = 1.0 - validation_scores
    train_error_rates_mean = np.mean(train_error_rates, axis=1)
    train_error_rates_std = np.std(train_error_rates, axis=1)
    validation_error_rates_mean = np.mean(validation_error_rates, axis=1)
    validation_error_rates_std = np.std(validation_error_rates, axis=1)

    if is_seq_in_range:
        new_param_range = []
        for ele in param_range:
            new_param_range.append(len(ele))
        param_range = new_param_range

    if is_iteration:
        # # plot iteration curves- accuracy
        # plot_lc_mcc(param_range,
        #             train_scores_mean, train_scores_std,
        #             validation_scores_mean, validation_scores_std,
        #             dataset_index, model_index, 4,
        #             y_lim[0], params, version + "_a",
        #             ('Accuracy', 'Iteration', 'Accuracy vs Iteration', 'iteration'),
        #             is_marked=False,
        #             is_fill_between=is_fill_between
        #             )
        # plot iteration curves- loss
        plot_lc_mcc(param_range,
                    train_error_rates_mean, train_error_rates_std,
                    validation_error_rates_mean, validation_error_rates_std,
                    dataset_index, model_index, 5,
                    y_lim[1], params, version + "_e",
                    ('Loss', 'Iteration', 'Loss vs Iteration', 'iteration'),
                    is_marked=False,
                    is_fill_between=is_fill_between
                    )
    else:
        # plot model complexity curves vs param_range - accuracy
        plot_lc_mcc(param_range,
                    train_scores_mean, train_scores_std,
                    validation_scores_mean, validation_scores_std,
                    dataset_index, model_index, 3,
                    y_lim[0], params, version + "_a",
                    ('Accuracy', param_name, 'Accuracy vs %s' % param_name, param_name),
                    is_marked=False,
                    is_fill_between=is_fill_between
                    )
        # # plot model complexity curves vs param_range - error rate
        # plot_lc_mcc(param_range,
        #             train_error_rates_mean, train_error_rates_std,
        #             validation_error_rates_mean, validation_error_rates_std,
        #             dataset_index, model_index, 3,
        #             y_lim[1], params, version + "_e",
        #             ('Error Rate', param_name, 'Error Rate vs %s' % param_name, param_name),
        #             is_marked=False,
        #             is_fill_between=is_fill_between
        #             )


def test_learning_model(classifier, data_dict):
    x_train = get_train_set_x(data_dict)
    y_train = get_train_set_y(data_dict)
    x_test = get_test_set_x(data_dict)
    y_test = get_test_set_y(data_dict)

    # train
    train_start = time.time()
    classifier.fit(x_train, y_train)
    train_end = time.time()

    # predict
    predict_start = time.time()
    train_predict = classifier.predict(x_train)
    test_predict = classifier.predict(x_test)
    predict_end = time.time()

    # accuracy
    train_score = accuracy_score(y_train, train_predict)
    test_score = accuracy_score(y_test, test_predict)

    # error rate
    train_error_rate = 1.0 - train_score
    test_error_rate = 1.0 - test_score

    # correlation
    train_correlation = np.corrcoef(train_predict, y=y_train.values.ravel())
    test_correlation = np.corrcoef(test_predict, y=y_test.values.ravel())

    # confusion matrix
    train_cm = confusion_matrix(y_train, train_predict).ravel()
    test_cm = confusion_matrix(y_test, test_predict).ravel()

    # recall
    train_recall = recall_score(y_train, train_predict).ravel()
    test_recall = recall_score(y_test, test_predict).ravel()

    # precision
    train_precision = precision_score(y_train, train_predict).ravel()
    test_precision = precision_score(y_test, test_predict).ravel()

    print("Train Results")
    print(f"Accuracy: {train_score}")
    print(f"Error Rate: {train_error_rate}")
    print(f"Confusion Matrix: {train_cm}")
    print(f"Correlation: {train_correlation[0, 1]}")
    print(f"Recall: {train_recall}")
    print(f"Precision: {train_precision}")
    print("Test Results")
    print(f"Accuracy: {test_score}")
    print(f"Error Rate: {test_error_rate}")
    print(f"Confusion Matrix: {test_cm}")
    print(f"Correlation: {test_correlation[0, 1]}")
    print(f"Recall: {test_recall}")
    print(f"Precision: {test_precision}")
    print("Time")
    print(f"Training Time: {train_end - train_start}")
    print(f"Prediction Time: {predict_end - predict_start}")

