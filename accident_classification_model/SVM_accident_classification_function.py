from accident_classification_model.accident_classification_data import x_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import os
from sklearn.externals import joblib


cv = 3

father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_document = result_directory + 'SVM accident classification.txt'
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
        f.write('C\t\tgamma\t\tkernel\n')
        f.write(str(history.best_params_['C']) + '\t\t' + str(history.best_params_['gamma']) + '\t\t\t'
                + history.best_params_['kernel'] + '\n')
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

        f.write('parameter_C\t')
        temp = history.cv_results_['param_C']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('parameter_gamma\t')
        temp = history.cv_results_['param_gamma']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('parameter_kernel\t')
        temp = history.cv_results_['param_kernel']
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


def svm_model(x_data_set, y_data_set, model_name, flag):
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)

    c_range = np.linspace(1, 20, 20)
    gamma_range = np.logspace(-2, 1, 20)
    parameter_distribution = {'kernel': ['rbf', 'sigmoid'], 'C': c_range, 'gamma': gamma_range}
    other_parameter = {'verbose': True}
    svm = SVC(**other_parameter)
    random_search = RandomizedSearchCV(svm, parameter_distribution, cv=cv, n_iter=50, verbose=2,
                                       return_train_score=True)
    history = random_search.fit(x_train, y_train)
    recode_best_parameter(history, flag)

    parameter_C = history.best_params_['C']
    parameter_gamma = history.best_params_['gamma']
    kernel = history.best_params_['kernel']
    print("SVM最佳参数搜索已完成...")

    svm = SVC(C=parameter_C, kernel=kernel, gamma=parameter_gamma)
    svm.fit(x_train, y_train)
    model_path = result_directory + '\\svm_model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    os.chdir(model_path)
    document_name = model_name + '_svm_model.m'
    if os.path.exists(document_name):
        os.remove(document_name)
    joblib.dump(svm, document_name)


def is_normal_svm_model():
    x_data_set, y_data_set = generate_data_set(0, [1, 2, 3, 4])
    svm_model(x_data_set, y_data_set, 'is_normal', 'w+')
    print("is_normal已完成...")


def prz_space_leak_svm_model():
    x_data_set, y_data_set = generate_data_set(1, [0, 2, 3, 4])
    svm_model(x_data_set, y_data_set, 'prz_space_leak', 'a')
    print("prz_space_leak已完成...")


def rcs_loca_knn_model():
    x_data_set, y_data_set = generate_data_set(2, [0, 1, 3, 4])
    svm_model(x_data_set, y_data_set, 'rcs_loca', 'a')
    print("rcs_loca已完成...")


def sg_2nd_side_leak_svm_model():
    x_data_set, y_data_set = generate_data_set(3, [0, 1, 2, 4])
    svm_model(x_data_set, y_data_set, 'sg_2nd_side_leak', 'a')
    print("sg_2nd_side_leak已完成...")


def sgtr_svm_model():
    x_data_set, y_data_set = generate_data_set(4, [0, 1, 2, 3])
    svm_model(x_data_set, y_data_set, 'sgtr', 'a')
    print("sgtr已完成...")
