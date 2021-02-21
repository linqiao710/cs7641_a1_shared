import numpy as np
import warnings
import sl_util
import config_util


def abalone_decision_tree_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.6, 1.05), (-0.05, 0.6), (0.0, 0.1)]
    y_lim_mcc_1 = [(0.55, 1.05), (-0.05, 0.5)]
    y_lim_mcc_2 = [(0.59, 0.74), (0.25, 0.41)]
    is_grid_search = False
    param_grid = [
        {'min_samples_leaf': [1], 'max_leaf_nodes': [69, 70, 71, 72]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_decision_tree_classifier(max_leaf_nodes=None, min_samples_leaf=1),
                                data_dict,
                                param_grid)
    else:
        # initial plot with default values
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_decision_tree_classifier(max_depth=100, max_leaf_nodes=850, criterion="entropy"),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=y_lim_lc,
            params="max_depth=100,max_leaf_nodes=850,criterion=entropy",
            version="0")

        # adjust 1st hyper-parameter: max_depth
        sl_util.generate_model_complexity_curves(
            sl_util.get_decision_tree_classifier(max_depth=100, max_leaf_nodes=850),
            param_name="max_depth",
            param_range=np.arange(1, 20, 1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="max_depth=[1, 19]",
            y_lim=[(0.7, 1.01), (-0.05, 0.5)],
            version="1")
        # plot after adjust 1st hyper-parameter: max_depth
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_decision_tree_classifier(max_depth=7, max_leaf_nodes=850, criterion="entropy"),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.7, 0.93), (-0.05, 0.6), (0.5, 0.8)],
            params="max_depth=7,max_leaf_nodes=850,criterion=entropy",
            version="1")

        # adjust 2nd hyper-parameter: max_leaf_nodes
        sl_util.generate_model_complexity_curves(
            sl_util.get_decision_tree_classifier(max_depth=7, max_leaf_nodes=850),
            param_name="max_leaf_nodes",
            param_range=np.arange(1, 50, 5),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="max_leaf_nodes=[1,45]",
            y_lim=[(0.75, 0.825), (-0.05, 0.5)],
            version="2")
        # plot after adjust 2nd hyper-parameter: max_leaf_nodes
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_decision_tree_classifier(max_depth=7, max_leaf_nodes=21, criterion="entropy"),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.7, 0.93), (-0.05, 0.6), (0.5, 0.8)],
            params="max_depth=7,max_leaf_nodes=21,criterion=entropy",
            version="2")

        print("Decision Tree before tuning")
        sl_util.test_learning_model(
            sl_util.get_decision_tree_classifier(max_depth=100, max_leaf_nodes=850, criterion="entropy"), data_dict)
        print("Decision Tree after tuning")
        sl_util.test_learning_model(
            sl_util.get_decision_tree_classifier(max_depth=7, max_leaf_nodes=21, min_samples_leaf=150,
                                                 criterion="entropy"), data_dict)


def wine_decision_tree_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.6, 1.05), (-0.05, 0.4), (0.0, 0.05)]
    y_lim_mcc_1 = [(0.65, 1.01), (-0.01, 0.4)]
    y_lim_mcc_2 = [(0.68, 0.875), (0.1, 0.35)]
    is_grid_search = False
    param_grid = [
        {'max_depth': [16], 'max_leaf_nodes': [69, 70, 71, 72]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_decision_tree_classifier(max_depth=None),
                                data_dict,
                                param_grid)
    else:
        # # initial plot with default values
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_decision_tree_classifier(max_depth=30, max_leaf_nodes=180, criterion="entropy"),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=y_lim_lc,
            params="max_depth=30,max_leaf_nodes=180,criterion=entropy",
            version="0")

        # adjust 1st hyper-parameter: max_depth
        sl_util.generate_model_complexity_curves(
            sl_util.get_decision_tree_classifier(max_depth=30, max_leaf_nodes=180, criterion="entropy"),
            param_name="max_depth",
            param_range=np.arange(2, 26, 2),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="max_depth=[1,24]",
            y_lim=[(0.69, 1.01), (-0.01, 0.4)],
            version="1")

        # plot after adjust 1st hyper-parameter: max_depth
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_decision_tree_classifier(max_depth=6, max_leaf_nodes=180, criterion="entropy"),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=y_lim_lc,
            params="max_depth=6,max_leaf_nodes=180,criterion=entropy",
            version="1")
        # #
        # adjust 2nd hyper-parameter: max_leaf_nodes
        sl_util.generate_model_complexity_curves(
            sl_util.get_decision_tree_classifier(max_depth=18, max_leaf_nodes=180, criterion="entropy"),
            param_name="max_leaf_nodes",
            param_range=np.arange(0, 180, 10),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="max_leaf_nodes=[1,170]",
            y_lim=[(0.7, 1.01), (0.1, 0.35)],
            version="2")
        # # plot after adjust 2nd hyper-parameter: max_leaf_nodes
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_decision_tree_classifier(max_depth=6, max_leaf_nodes=20, criterion="entropy"),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=y_lim_lc,
            params="max_depth=6,max_leaf_nodes=20,criterion=entropy",
            version="2")

        print("Decision Tree before tuning")
        sl_util.test_learning_model(
            sl_util.get_decision_tree_classifier(max_depth=30, max_leaf_nodes=180), data_dict)
        print("Decision Tree after tuning")
        sl_util.test_learning_model(
            sl_util.get_decision_tree_classifier(max_depth=6, max_leaf_nodes=20), data_dict)


def abalone_neural_networks_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.1, 1.05), (-0.05, 0.6), (0.0, 0.04)]
    y_lim_iteration = [(0.2, 0.8), (0.2, 0.8)]
    y_lim_mcc = [(0.55, 0.80), (0.25, 0.45)]
    is_grid_search = False
    param_grid = [
        {'momentum': [0.9], 'learning_rate_init': [0.001]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_neural_network_classifier(),
                                data_dict,
                                param_grid)
    else:

        # # initial plot with default values
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(1, ), learning_rate_init=0.001, max_iter=600),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.65, 0.8), (0.4,0.6), (0.0, 0.04)],
            params="hidden_layer_sizes=(1,),learning_rate_init=0.001,max_iter=600",
            version="0", is_fill_between=False)
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(1, ), learning_rate_init=0.001),
            param_name="max_iter",
            param_range=np.arange(0, 1050, 50),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="hidden_layer_sizes=(1,),learning_rate_init=0.001",
            y_lim=[(0.2, 0.5), (0.2, 0.51)],
            version="0",
            is_fill_between=False,
            is_iteration=True)

        # adjust 1st hyper-parameter: hidden_layer_sizes
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(1,), learning_rate_init=0.001),
            param_name="hidden_layer_sizes",
            param_range=list((10,) * ele for ele in range(1, 11, 2)),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="hidden_layer_sizes=[(10,)*n]",
            y_lim=[(0.5, 0.77), (0.25, 0.45)],
            version="1",
            is_fill_between=False, is_seq_in_range=True)

        # plot after adjust 1st hyper-parameter: hidden_layer_sizes
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,)*2, learning_rate_init=0.001, max_iter=360),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.7, 0.8), (-0.05, 0.6), (0.0, 0.04)],
            params="hidden_layer_sizes=(10,)*2,learning_rate_init=0.001,\nmax_iter=360",
            version="1", is_fill_between=False)
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,)*2, learning_rate_init=0.001),
            param_name="max_iter",
            param_range=np.arange(1, 551, 50),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="hidden_layer_sizes=(10,)*2,learning_rate_init=0.001",
            y_lim=[(0.2, 0.68), (0.19, 0.35)],
            version="1",
            is_fill_between=False,
            is_iteration=True)

        # adjust 2nd hyper-parameter: learning_rate_init
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,)*2, learning_rate_init=0.001),
            param_name="learning_rate_init",
            param_range=np.arange(0.0001, 0.0071, 0.0005),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="learning_rate_init=[0.0001, 0.0066]",
            y_lim=[(0.72, 0.82), (0.20, 0.40)],
            version="2")

        # plot after adjust 2nd hyper-parameter: learning_rate_init
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,) * 2, learning_rate_init=0.0011,
                                                  max_iter=350),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.7, 0.8), (-0.05, 0.6), (0.0, 0.04)],
            params="hidden_layer_sizes=(10,)*2,learning_rate_init=0.0011,\nmax_iter=350",
            version="2", is_fill_between=False)
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,) * 2, learning_rate_init=0.0011),
            param_name="max_iter",
            param_range=np.arange(1, 551, 50),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="hidden_layer_sizes=(10,)*2, learning_rate_init=0.0011",
            y_lim=[(0.62, 0.68), (0.19, 0.35)],
            version="2",
            is_fill_between=False,
            is_iteration=True)

        print("Neural Networks before tuning")
        sl_util.test_learning_model(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(1,), learning_rate_init=0.001, max_iter=600),
            data_dict)
        print("Neural Networks after tuning")
        sl_util.test_learning_model(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,) * 2, learning_rate_init=0.0011,
                                                  max_iter=350), data_dict)


def wine_neural_networks_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.4, 0.9), (-0.05, 0.6), (0.0, 0.04)]
    y_lim_iteration = [(0.3, 0.8), (0.20, 0.7)]
    y_lim_mcc = [(0.6, 0.80), (0.20, 0.40)]
    is_grid_search = False
    param_grid = [
        {'momentum': [0.9], 'learning_rate_init': [0.001]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_neural_network_classifier(),
                                data_dict,
                                param_grid)
    else:

        # initial plot with default values
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(1,), learning_rate_init=0.001, max_iter=500),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.55, 0.75), (0.2, 0.5), (0.0, 0.04)],
            params="hidden_layer_sizes=(1,),learning_rate_init=0.001,\nmax_iter=500",
            version="0", is_fill_between=False)
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(1,), learning_rate_init=0.001),
            param_name="max_iter",
            param_range=np.arange(1, 951, 50),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="hidden_layer_sizes=(1,), learning_rate_init=0.001",
            y_lim=[(0.3, 0.8), (0.25, 0.4)],
            version="0",
            is_fill_between=False,
            is_iteration=True)

        # adjust 1st hyper-parameter: hidden_layer_sizes
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(),
            param_name="hidden_layer_sizes",
            param_range=list((10,) * ele for ele in range(1, 6, 1)),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="hidden_layer_sizes=[(10,)*n]",
            y_lim=[(0.5, 0.8), (0.25, 0.45)],
            version="1",
            is_fill_between=False, is_seq_in_range=True)

        # plot after adjust 1st hyper-parameter: hidden_layer_sizes
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,), learning_rate_init=0.001, max_iter=400),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.7, 0.78), (-0.05, 0.6), (0.0, 0.04)],
            params="hidden_layer_sizes=(10,),learning_rate_init=0.001,\nmax_iter=400",
            version="1", is_fill_between=False)
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,), learning_rate_init=0.001),
            param_name="max_iter",
            param_range=np.arange(1, 951, 50),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="hidden_layer_sizes=(10,),learning_rate_init=0.001",
            y_lim=[(0.2, 0.5), (0.20, 0.60)],
            version="1",
            is_fill_between=False,
            is_iteration=True)

        # adjust 2nd hyper-parameter: learning_rate_init
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,), learning_rate_init=0.001),
            param_name="learning_rate_init",
            param_range=np.arange(0.0001, 0.0011, 0.0001),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="learning_rate_init=[0.0001, 0.001]",
            y_lim=[(0.55, 0.8), (0.20, 0.40)],
            version="2")

        # plot after adjust 2nd hyper-parameter: learning_rate_init
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,), learning_rate_init=0.0007, max_iter=410),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.68, 0.78), (-0.05, 0.6), (0.0, 0.04)],
            params="hidden_layer_sizes=(10,),learning_rate_init=0.0007,\nmax_iter=410",
            version="2", is_fill_between=False)
        sl_util.generate_model_complexity_curves(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,), learning_rate_init=0.0008),
            param_name="max_iter",
            param_range=np.arange(1, 951, 50),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="hidden_layer_sizes=(10,),learning_rate_init=0.0008",
            y_lim=[(0.2, 0.80), (0.20, 0.55)],
            version="2",
            is_fill_between=False,
            is_iteration=True)

        print("Neural Networks before tuning")
        sl_util.test_learning_model(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(1,), learning_rate_init=0.001, max_iter=500),
            data_dict)
        print("Neural Networks after tuning")
        sl_util.test_learning_model(
            sl_util.get_neural_network_classifier(hidden_layer_sizes=(10,), learning_rate_init=0.0007, max_iter=410),
            data_dict)


def abalone_boosting_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.5, 1.05), (-0.05, 0.6), (-0.01, 3.0)]
    y_lim_mcc = [(0.50, 0.8), (0.2, 0.5)]
    is_grid_search = False
    param_grid = [
        {'n_estimators': [60]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_adaboost_classifier(),
                                data_dict,
                                param_grid)
    else:
        # initial plot with default values
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_adaboost_classifier(n_estimators=1, learning_rate=1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim= [(0.7, 1.05), (-0.05, 0.6), (-0.01, 3.0)],
            params="n_estimators=1,learning_rate=1",
            version="0")

        # adjust 1st hyper-parameter: n_estimators
        sl_util.generate_model_complexity_curves(
            sl_util.get_adaboost_classifier(n_estimators=1, learning_rate=1),
            param_name="n_estimators",
            param_range=np.arange(1, 800, 100),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="n_estimators=[1, 700]",
            y_lim=[(0.75, 0.85), (0.2, 0.5)],
            version="1")

        # plot after adjust 1st hyper-parameter: n_estimators
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_adaboost_classifier(n_estimators=400, learning_rate=1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.7, 1.05), (-0.05, 0.6), (-0.01, 3.0)],
            params="n_estimators=400,learning_rate=1",
            version="1")

        # adjust 2nd hyper-parameter: learning_rate
        sl_util.generate_model_complexity_curves(
            sl_util.get_adaboost_classifier(n_estimators=400),
            param_name="learning_rate",
            param_range=np.arange(0.01, 1.21, 0.3),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="learning_rate=[0.01, 1]",
            y_lim=[(0.75, 0.84), (0.2, 0.5)],
            version="2")

        # plot after adjust 2nd hyper-parameter: learning_rate
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_adaboost_classifier(n_estimators=400, learning_rate=0.3),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.7, 1.05), (-0.05, 0.6), (-0.01, 3.0)],
            params="n_estimators=400,learning_rate=0.3",
            version="2")

        print("Boosting before tuning")
        sl_util.test_learning_model(
            sl_util.get_adaboost_classifier(n_estimators=1000, learning_rate=1), data_dict)
        print("Boosting after tuning")
        sl_util.test_learning_model(
            sl_util.get_adaboost_classifier(n_estimators=400, learning_rate=0.3), data_dict)


def wine_boosting_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.6, 1.05), (-0.05, 0.6), (-0.01, 1.0)]
    y_lim_mcc = [(0.65, 0.85), (0.15, 0.35)]
    is_grid_search = False
    param_grid = [
        {'n_estimators': [60]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_adaboost_classifier(),
                                data_dict,
                                param_grid)
    else:

        # initial plot with default values
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_adaboost_classifier(n_estimators=3500, learning_rate=1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=y_lim_lc,
            params="n_estimators=3500,learning_rate=1",
            version="0")

        # adjust 1st hyper-parameter: n_estimators
        sl_util.generate_model_complexity_curves(
            sl_util.get_adaboost_classifier(n_estimators=1000,learning_rate=1),
            param_name="n_estimators",
            param_range=np.arange(1, 451, 50),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="n_estimators=[1, 450]",
            y_lim=[(0.65, 0.95), (0.15, 0.35)],
            version="1")

        # plot after adjust 1st hyper-parameter: n_estimators
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_adaboost_classifier(n_estimators=250, learning_rate=1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=y_lim_lc,
            params="n_estimators=250, learning_rate=1",
            version="1")

        # adjust 2nd hyper-parameter: learning_rate
        sl_util.generate_model_complexity_curves(
            sl_util.get_adaboost_classifier(n_estimators=250),
            param_name="learning_rate",
            param_range=np.arange(0.01, 1.11, 0.1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="learning_rate=[0.01, 1]",
            y_lim=[(0.7, 0.9), (0.15, 0.35)],
            version="2")

        # plot after adjust 2nd hyper-parameter: learning_rate
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_adaboost_classifier(n_estimators=250, learning_rate=0.39),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=y_lim_lc,
            params="n_estimators=250, learning_rate=0.39",
            version="2")

        print("Boosting before tuning")
        sl_util.test_learning_model(
            sl_util.get_adaboost_classifier(n_estimators=3500, learning_rate=1), data_dict)
        print("Boosting after tuning")
        sl_util.test_learning_model(
            sl_util.get_adaboost_classifier(n_estimators=250, learning_rate=0.39), data_dict)


def abalone_svm_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.2, 0.75), (0.2, 0.7), (-0.03, 0.6)]
    y_lim_mcc = [(0.3, 0.8), (0.2, 0.7)]
    is_grid_search = False
    param_grid = [
        {"kernel": ["linear", "poly", "rbf"]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_adaboost_classifier(),
                                data_dict,
                                param_grid)
    else:

        # initial plots with 3 kernels ["rbf", "linear", "poly"]
        kernel_values = ["rbf", "linear", "poly"]
        for index in range(len(kernel_values)):
            sl_util.generate_learning_curves_sizes_times(
                sl_util.get_svm_classifier(kernel=kernel_values[index], max_iter=1),
                data_dict=data_dict,
                dataset_index=dataset_index,
                model_index=model_index,
                y_lim=y_lim_lc,
                params="kernel=%s" % kernel_values[index],
                version="0_%s" % (kernel_values[index]))

        # linear: adjust 1st hyper-parameter: max_iter
        sl_util.generate_model_complexity_curves(
            sl_util.get_svm_classifier(kernel="linear"),
            param_name="max_iter",
            param_range=np.arange(500, 5000, 500),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params='kernel=linear,max_iter=[500,4500]',
            y_lim=[(0.65, 0.82), (0.2, 0.7)],
            version="1_linear")

        # linear: plot after adjust 1st hyper-parameter: max_iter
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_svm_classifier(kernel="linear", max_iter=3000),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.75, 0.82), (0.2, 0.7), (-0.03, 0.6)],
            params="kernel=linear, max_iter=3000",
            version="1_linear")

        # rbf: adjust 1st hyper-parameter: max_iter
        sl_util.generate_model_complexity_curves(
            sl_util.get_svm_classifier(kernel="rbf"),
            param_name="max_iter",
            param_range=np.arange(500, 1600, 200),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="kernel=rbf,max_iter=[500,1400]",
            y_lim=[(0.65, 0.82), (0.2, 0.7)],
            version="1_rbf")

        # rbf: plot after adjust 1st hyper-parameter: max_iter
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_svm_classifier(kernel="rbf", max_iter=1000),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.74, 0.81), (0.2, 0.7), (-0.03, 0.6)],
            params="kernel=rbf, max_iter=1000",
            version="1_rbf")

        print("SVM-linear before tuning")
        sl_util.test_learning_model(
            sl_util.get_svm_classifier(kernel="linear", max_iter=1), data_dict)
        print("SVM-linear after tuning")
        sl_util.test_learning_model(
            sl_util.get_svm_classifier(kernel="linear", max_iter=3000), data_dict)
        print("SVM-rbf before tuning")
        sl_util.test_learning_model(
            sl_util.get_svm_classifier(kernel="rbf", max_iter=1), data_dict)
        print("SVM-rbf after tuning")
        sl_util.test_learning_model(
            sl_util.get_svm_classifier(kernel="rbf", max_iter=1000), data_dict)


def wine_svm_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.35, 0.65), (0.1, 0.6), (-0.03, 0.6)]
    y_lim_mcc = [(0.4, 0.9), (0.1, 0.6)]
    is_grid_search = False
    param_grid = [
        {"kernel": ["linear", "poly", "rbf"]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_adaboost_classifier(),
                                data_dict,
                                param_grid)
    else:
        # initial plots with 3 kernels ["rbf", "linear", "poly"]
        kernel_values = ["rbf", "linear", "poly"]
        for index in range(len(kernel_values)):
            sl_util.generate_learning_curves_sizes_times(
                sl_util.get_svm_classifier(kernel=kernel_values[index], max_iter=1),
                data_dict=data_dict,
                dataset_index=dataset_index,
                model_index=model_index,
                y_lim=y_lim_lc,
                params="kernel=%s" % kernel_values[index],
                version="0_%s" % (kernel_values[index]))

        # linear: adjust 1st hyper-parameter: max_iter
        sl_util.generate_model_complexity_curves(
            sl_util.get_svm_classifier(kernel="linear"),
            param_name="max_iter",
            param_range=np.arange(1, 3601, 300),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params='kernel=linear,max_iter=[1,3300]',
            y_lim=[(0.6, 0.775), (0.2, 0.7)],
            version="1_linear")

        # linear: plot after adjust 1st hyper-parameter: max_iter
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_svm_classifier(kernel="linear", max_iter=1500),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.7, 0.825), (0.2, 0.7), (-0.03, 0.6)],
            params="kernel=linear, max_iter=1500",
            version="1_linear")

        # rbf: adjust 1st hyper-parameter: max_iter
        sl_util.generate_model_complexity_curves(
            sl_util.get_svm_classifier(kernel="rbf", max_iter=1),
            param_name="max_iter",
            param_range=np.arange(0, 1000, 100),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params='kernel=rbf, max_iter=[0,900]',
            y_lim=[(0.5, 0.81), (0.2, 0.7)],
            version="1_rbf")

        # rbf: plot after adjust 1st hyper-parameter: max_iter
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_svm_classifier(kernel="rbf", max_iter=400),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.6, 0.95), (0.2, 0.7), (-0.03, 0.6)],
            params="kernel=rbf, max_iter=400",
            version="1_rbf")

        # poly: adjust 2nd hyper-parameter: degree
        sl_util.generate_model_complexity_curves(
            sl_util.get_svm_classifier(kernel="poly"),
            param_name="degree",
            param_range=np.arange(1, 6, 2),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params='kernel=poly',
            y_lim=[(0.7, 0.8), (0.2, 0.7)],
            version="2_poly")
        # poly: plot after adjust 2nd hyper-parameter: degree
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_svm_classifier(kernel="poly", max_iter=800, degree=1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.6, 0.95), (0.2, 0.7), (-0.03, 0.6)],
            params="kernel=poly, max_iter=800, degree=1",
            version="2_poly")

        print("SVM-linear before tuning")
        sl_util.test_learning_model(
            sl_util.get_svm_classifier(kernel="linear", max_iter=1), data_dict)
        print("SVM-linear after tuning")
        sl_util.test_learning_model(
            sl_util.get_svm_classifier(kernel="linear", max_iter=1500), data_dict)

        print("SVM-rbf before tuning")
        sl_util.test_learning_model(
            sl_util.get_svm_classifier(kernel="rbf", max_iter=1), data_dict)
        print("SVM-rbf after tuning")
        sl_util.test_learning_model(
            sl_util.get_svm_classifier(kernel="rbf", max_iter=400), data_dict)


def abalone_knn_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.65, 1.05), (0.1, 0.6), (-0.03, 0.1)]
    y_lim_mcc = [(0.5, 1.01), (-0.01, 0.5)]
    is_grid_search = False
    param_grid = [
        {"n_neighbors": [3, 10, 20]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_knn_classifier(),
                                data_dict,
                                param_grid)
    else:
        # initial plot with default values
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_knn_classifier(n_neighbors=1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=y_lim_lc,
            params="n_neighbors=1",
            version="0")

        # adjust 1st hyper-parameter: n_neighbors
        sl_util.generate_model_complexity_curves(
            sl_util.get_knn_classifier(),
            param_name="n_neighbors",
            param_range=np.arange(10, 110, 10),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="n_neighbors=[10,100]",
            y_lim=[(0.76, 0.82), (-0.01, 0.5)],
            version="1")

        # plot after adjust 1st hyper-parameter: n_neighbors
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_knn_classifier(n_neighbors=40),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.72, 0.82), (0.1, 0.6), (-0.03, 0.1)],
            params="n_neighbors=40",
            version="1")
        #

        print("KNN before tuning")
        sl_util.test_learning_model(
            sl_util.get_knn_classifier(n_neighbors=1), data_dict)
        print("KNN after tuning")
        sl_util.test_learning_model(
            sl_util.get_knn_classifier(n_neighbors=40), data_dict)


def wine_knn_analysis(data_dict, dataset_index, model_index):
    y_lim_lc = [(0.5, 0.9), (0.1, 0.6), (-0.03, 0.1)]
    y_lim_mcc = [(0.6, 1.01), (-0.01, 0.5)]
    is_grid_search = False
    param_grid = [
        {"n_neighbors": [3, 10, 20]}
    ]

    # grid search
    if is_grid_search:
        sl_util.exe_grid_search(sl_util.get_knn_classifier(),
                                data_dict,
                                param_grid)
    else:
        # initial plot with default values
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_knn_classifier(n_neighbors=1),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.6, 1.05), (0.1, 0.6), (-0.03, 0.1)],
            params="n_neighbors=1",
            version="0")

        # adjust 1st hyper-parameter: n_neighbors
        sl_util.generate_model_complexity_curves(
            sl_util.get_knn_classifier(),
            param_name="n_neighbors",
            param_range=np.arange(1, 121, 20),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            params="n_neighbors=[1,100]",
            y_lim=[(0.72, 1.01), (-0.01, 0.5)],
            version="1")

        # plot after adjust 1st hyper-parameter: n_neighbors
        sl_util.generate_learning_curves_sizes_times(
            sl_util.get_knn_classifier(n_neighbors=63),
            data_dict=data_dict,
            dataset_index=dataset_index,
            model_index=model_index,
            y_lim=[(0.6, 0.8), (0.1, 0.6), (-0.03, 0.1)],
            params="n_neighbors=63",
            version="1")

        print("KNN before tuning")
        sl_util.test_learning_model(
            sl_util.get_knn_classifier(n_neighbors=1), data_dict)
        print("KNN after tuning")
        sl_util.test_learning_model(
            sl_util.get_knn_classifier(n_neighbors=63), data_dict)


def sl_algorithms(data_dict, dataset_index):
    if dataset_index == 0:
        abalone_decision_tree_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=0)
        abalone_neural_networks_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=1)
        abalone_boosting_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=2)
        abalone_svm_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=3)
        abalone_knn_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=4)
    elif dataset_index == 1:
        wine_decision_tree_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=0)
        wine_neural_networks_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=1)
        wine_boosting_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=2)
        wine_svm_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=3)
        wine_knn_analysis(data_dict=data_dict, dataset_index=dataset_index, model_index=4)


if __name__ == "__main__":
    np.random.seed(config_util.ran_state)
    warnings.filterwarnings("ignore")

    abalone_data = sl_util.load_data("abalone_a1_v3.csv")
    abalone_data = sl_util.shuffle_df(abalone_data, 5)
    abalone_data_dict = sl_util.split_train_test(abalone_data)
    sl_util.scale_attributes(data_dict=abalone_data_dict)
    sl_algorithms(data_dict=abalone_data_dict, dataset_index=0)

    wine_quality_data = sl_util.load_data("wine_quality_red_a1.csv")
    wine_quality_data = sl_util.shuffle_df(wine_quality_data, 5)
    wine_quality_data_dict = sl_util.split_train_test(wine_quality_data)
    sl_util.print_description(data_dict=wine_quality_data_dict, dataset_index=1, pre_text="Before")
    sl_util.scale_attributes(data_dict=wine_quality_data_dict)
    sl_util.print_description(data_dict=wine_quality_data_dict, dataset_index=1, pre_text="After")
    sl_algorithms(data_dict=wine_quality_data_dict, dataset_index=1)

    # mean, std etc
    sl_util.print_dataset_info(abalone_data, 0)
    sl_util.print_dataset_info(wine_quality_data, 1)

    # percentage of labels in output
    print("Abalone Output: 'is_mature'")
    print(sl_util.get_output_distribution(sl_util.get_dataset_y(abalone_data)))
    print("Wine Quality Output: 'is_good_quality'")
    print(sl_util.get_output_distribution(sl_util.get_dataset_y(wine_quality_data)))

    pass
