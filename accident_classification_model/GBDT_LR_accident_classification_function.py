from accident_classification_model.accident_classification_data import x_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import os


cv = 3
father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_document = result_directory + 'GBDT+LR accident classification.txt'
if os.path.exists(result_document):
    os.remove(result_document)


def GridSearch_result(history, flag):
    with open(result_document, flag) as f:
        f.write('n_estimators\t\tlearning_rate\n')
        f.write(str(history.best_params_['n_estimators']) + '\t\t' + str(history.best_params_['learning_rate']) + '\n')
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

        f.write('parameter_learning_rate\t')
        temp = history.cv_results_['param_learning_rate']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('parameter_n_estimators\t')
        temp = history.cv_results_['param_n_estimators']
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


def GBDT_data_set(index, index_list) -> object:
    x_train = x_data[index]
    y_train_normal = np.full(len(x_data[index]), 1)
    abnormal_set_length = 0

    for i_value in index_list:
        x_train = np.vstack((x_train, x_data[i_value]))
        abnormal_set_length = abnormal_set_length + len(x_data[index])

    y_train_abnormal = np.zeros(abnormal_set_length)
    y_train = np.hstack((y_train_normal, y_train_abnormal))
    return x_train, y_train


def GBDT_model(x_data_set, y_data_set, flag):
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)
    parameter_distribution = {'n_estimators': np.arange(50, 100, 10),
                              'learning_rate': np.logspace(-3, 1, 10)}
    gbdt_model = GradientBoostingClassifier()
    grid_search = GridSearchCV(gbdt_model, parameter_distribution, cv=cv, verbose=2, return_train_score=True)
    history = grid_search.fit(x_train, y_train)
    GridSearch_result(history, flag)

    best_estimators = history.best_params_['n_estimators']
    best_learning_rate = history.best_params_['learning_rate']
    gbdt_best = GradientBoostingClassifier(n_estimators=best_estimators, learning_rate=best_learning_rate)

    gbdt_best.fit(x_train, y_train)
    gbdt_feature = gbdt_best.apply(x_train)
    gbdt_feature = gbdt_feature.reshape(-1, best_estimators)

    enc = OneHotEncoder()
    enc.fit(gbdt_feature)

    gbdt_best_feature = np.array(enc.transform(gbdt_feature).toarray())
    lr = LogisticRegression()
    lr.fit(gbdt_best_feature, y_train)


def GBDT_is_normal_model():
    x_train, y_train = GBDT_data_set(0, [1, 2, 3, 4])
    GBDT_model(x_train, y_train, "w+")


def GBDT_prz_space_leak_model():
    x_train, y_train = GBDT_data_set(1, [0, 2, 3, 4])
    GBDT_model(x_train, y_train, "a")


def GBDT_rcs_loca_model():
    x_train, y_train = GBDT_data_set(2, [0, 1, 3, 4])
    GBDT_model(x_train, y_train, "a")


def GBDT_sg_2nd_side_leak_model():
    x_train, y_train = GBDT_data_set(3, [0, 1, 2, 4])
    GBDT_model(x_train, y_train, "a")


def GBDT_sgtr_model():
    x_train, y_train = GBDT_data_set(4, [0, 1, 2, 3])
    GBDT_model(x_train, y_train, "a")
