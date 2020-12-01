from accident_classification_model.accident_classification_data import x_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import os
from sklearn.externals import joblib


cv = 3
neighbors = 11

father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_document = result_directory + 'kNN accident classification.txt'
if os.path.exists(result_document):
    os.remove(result_document)


def generate_data_set(index, index_list) -> object:
    x_data_set = x_data[index]
    length = 0
    for i_value in index_list:
        x_data_set = np.vstack((x_data_set, x_data[i_value]))
        length = length + len(x_data[i_value])
    y_data_set = np.hstack((np.full(len(x_data[index]), 1), np.zeros(length)))

    return x_data_set, y_data_set


def recode_best_parameter(history, flag):
    with open(result_document, flag) as f:
        f.write('algorithm\t\tn_neighbors\t\tweights\n')
        f.write(history.best_params_['algorithm'] + '\t\t' + str(history.best_params_['n_neighbors']) + '\t\t\t'
                + history.best_params_['weights'] + '\n')
        f.write('best_params\t' + str(history.best_score_) + '\n')

        f.write('mean_fit_time\t')
        temp = history.cv_results_['mean_fit_time']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('std_fit_time\t')
        temp = history.cv_results_['std_fit_time']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('mean_score_time\t')
        temp = history.cv_results_['mean_score_time']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('std_score_time\t')
        temp = history.cv_results_['std_score_time']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('parameter_algorithm\t')
        temp = history.cv_results_['param_algorithm']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('parameter_n_neighbors\t')
        temp = history.cv_results_['param_n_neighbors']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('parameter_weights\t')
        temp = history.cv_results_['param_weights']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('mean_test_score\t')
        temp = history.cv_results_['mean_test_score']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('std_test_score\t')
        temp = history.cv_results_['std_test_score']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('mean_train_score\t')
        temp = history.cv_results_['mean_train_score']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('std_train_score\t')
        temp = history.cv_results_['std_train_score']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')


def knn_model(x_data_set, y_data_set, model_name, flag):
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)
    parameter_distribution = {'n_neighbors': [i for i in np.arange(1, neighbors)],
                              'weights': ['uniform', 'distance'],
                              'algorithm': ['auto', 'kd_tree']}

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, parameter_distribution, cv=cv, verbose=2, return_train_score=True)
    history = grid_search.fit(x_train, y_train)
    recode_best_parameter(history, flag)

    knn_best = KNeighborsClassifier(n_neighbors=history.best_params_['n_neighbors'],
                                    algorithm=history.best_params_['algorithm'],
                                    weights=history.best_params_['weights'])

    knn_best.fit(x_train, y_train)
    model_path = result_directory + '\\knn_model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    os.chdir(model_path)
    document_name = model_name + '_knn_model.m'
    if os.path.exists(model_path + document_name):
        os.remove(document_name)
    joblib.dump(knn_best, document_name)


def is_normal_knn_model():
    x_data_set, y_data_set = generate_data_set(0, [1, 2, 3, 4])
    knn_model(x_data_set, y_data_set, 'is_normal', 'w+')


def prz_space_leak_knn_model():
    x_data_set, y_data_set = generate_data_set(1, [0, 2, 3, 4])
    knn_model(x_data_set, y_data_set, 'prz_space_leak', 'a')


def rcs_loca_knn_model():
    x_data_set, y_data_set = generate_data_set(2, [0, 1, 3, 4])
    knn_model(x_data_set, y_data_set, 'rcs_loca', 'a')


def sg_2nd_side_leak_knn_model():
    x_data_set, y_data_set = generate_data_set(3, [0, 1, 2, 4])
    knn_model(x_data_set, y_data_set, 'sg_2nd_side_leak', 'a')


def sgtr_knn_model():
    x_data_set, y_data_set = generate_data_set(4, [0, 1, 2, 3])
    knn_model(x_data_set, y_data_set, 'sgtr', 'a')
