from accident_classification_model.accident_classification_data import x_data
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import xgboost
import time
from collections import Counter

father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

ECOC_matrix = np.array([[1, -1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, -1, 1, -1, -1], [-1, -1, -1, 1, -1],
                        [-1, -1, -1, -1, 1]])
x_test_set = np.vstack((x_data[0][:2000, ], x_data[1][:2000, ], x_data[2][:2000, ], x_data[3][:2000, ],
                        x_data[4][:2000, ]))


def generate_data_set(index, index_list) -> object:
    x_data_set = x_data[index]
    length = 0
    for i_value in index_list:
        x_data_set = np.vstack((x_data_set, x_data[i_value]))
        length = length + len(x_data[i_value])
    y_data_set = np.hstack((np.full(len(x_data[index]), 1), np.zeros(length)))

    return x_data_set, y_data_set


def record_label(label_data, label_name, flag):
    result_document = result_directory + 'label_result.txt'
    length = len(label_data)
    with open(result_document, flag) as f:
        f.write(label_name + "\n")
        for i_value in np.arange(length):
            if (i_value + 1) % 100 != 0 and i_value != (length - 1):
                f.write(str(label_data[i_value]) + ",")
            else:
                f.write(str(label_data[i_value]) + "\n")


def accident_classification(y_predict) -> object:
    y_predict = np.transpose(y_predict)
    column_length = y_predict.shape[0]
    row_length = y_predict.shape[1]
    y_label = []

    for i_value in np.arange(column_length):
        for j_value in np.arange(row_length):
            if y_predict[i_value][j_value] == 0:
                y_predict[i_value][j_value] = -1

    for i_value in np.arange(column_length):
        distance = []
        for j_value in np.arange(ECOC_matrix.shape[0]):
            temp_distance = 0
            for k_value in np.arange(ECOC_matrix.shape[1]):
                temp_distance = temp_distance + (y_predict[i_value][k_value] - ECOC_matrix[j_value][k_value]) ** 2
            distance.append(np.sqrt(temp_distance))

        min_distance = min(distance)
        index = distance.index(min_distance)
        y_label.append(index)

    return y_label


def knn_single_model(x_data_set, y_data_set, n_neighbors, weights, algorithm) -> object:
    train_start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    knn.fit(x_train, y_train)
    train_end_time = time.time()

    train_time = train_end_time - train_start_time

    predict_start_time = time.time()
    y_predict = knn.predict(x_test_set)
    predict_end_time = time.time()

    predict_time = predict_end_time - predict_start_time

    return y_predict, [train_time, predict_time]


def knn_predict_model():
    y_data_set = np.hstack((np.full(5000, 1), np.zeros(20000)))

    x_data_set = np.vstack((x_data[0], x_data[1], x_data[2], x_data[3], x_data[4]))
    y_predict_is_normal, time_is_normal = knn_single_model(x_data_set, y_data_set, 1, 'uniform', 'auto')

    x_data_set = np.vstack((x_data[1], x_data[0], x_data[2], x_data[3], x_data[4]))
    y_predict_prz_space_leak, time_prz_space_leak = knn_single_model(x_data_set, y_data_set, 9, 'distance', 'auto')

    x_data_set = np.vstack((x_data[2], x_data[0], x_data[1], x_data[3], x_data[4]))
    y_predict_rcs_loca, time_rcs_loca = knn_single_model(x_data_set, y_data_set, 1, 'uniform', 'auto')

    x_data_set = np.vstack((x_data[3], x_data[0], x_data[1], x_data[2], x_data[4]))
    y_predict_sg_2nd_side_leak, time_sg_2nd_side_leak = knn_single_model(x_data_set, y_data_set, 8, 'uniform', 'auto')

    x_data_set = np.vstack((x_data[4], x_data[0], x_data[1], x_data[2], x_data[3]))
    y_predict_sgtr, time_sgtr = knn_single_model(x_data_set, y_data_set, 1, 'uniform', 'auto')

    y_predict = np.vstack((y_predict_is_normal, y_predict_prz_space_leak, y_predict_rcs_loca,
                           y_predict_sg_2nd_side_leak, y_predict_sgtr))

    y_label = accident_classification(y_predict)
    train_time = time_is_normal[0] + time_prz_space_leak[0] + time_rcs_loca[0] + time_sg_2nd_side_leak[0] + time_sgtr[0]
    pre_time = time_is_normal[1] + time_prz_space_leak[1] + time_rcs_loca[1] + time_sg_2nd_side_leak[1] + time_sgtr[1]

    print('knn\t' + 'train_time\t' + str(train_time) + '\t' + 'predict_time\t' + str(pre_time))

    return y_label, [train_time, pre_time]


def svm_single_model(x_data_set, y_data_set, parameter_c, kernel, parameter_gamma) -> object:
    train_start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)
    svm = SVC(C=parameter_c, kernel=kernel, gamma=parameter_gamma, verbose=True)
    svm.fit(x_train, y_train)
    train_end_time = time.time()

    train_time = train_end_time - train_start_time

    predict_start_time = time.time()
    y_predict = svm.predict(x_test_set)
    predict_end_time = time.time()

    predict_time = predict_end_time - predict_start_time

    return y_predict, [train_time, predict_time]


def svm_predict_model():
    y_data_set = np.hstack((np.full(5000, 1), np.zeros(20000)))

    x_data_set = np.vstack((x_data[0], x_data[1], x_data[2], x_data[3], x_data[4]))
    y_predict_is_normal, time_is_normal = svm_single_model(x_data_set, y_data_set, 17, 'rbf', 0.0144)

    x_data_set = np.vstack((x_data[1], x_data[0], x_data[2], x_data[3], x_data[4]))
    y_predict_prz_space_leak, time_prz_space_leak = svm_single_model(x_data_set, y_data_set, 14, 'rbf', 0.5456)

    x_data_set = np.vstack((x_data[2], x_data[0], x_data[1], x_data[3], x_data[4]))
    y_predict_rcs_loca, time_rcs_loca = svm_single_model(x_data_set, y_data_set, 20, 'rbf', 0.1274)

    x_data_set = np.vstack((x_data[3], x_data[0], x_data[1], x_data[2], x_data[4]))
    y_predict_sg_2nd_side_leak, time_sg_2nd_side_leak = svm_single_model(x_data_set, y_data_set, 7, 'rbf', 0.0207)

    x_data_set = np.vstack((x_data[4], x_data[0], x_data[1], x_data[2], x_data[3]))
    y_predict_sgtr, time_sgtr = svm_single_model(x_data_set, y_data_set, 9, 'rbf', 0.01)

    y_predict = np.vstack((y_predict_is_normal, y_predict_prz_space_leak, y_predict_rcs_loca,
                           y_predict_sg_2nd_side_leak, y_predict_sgtr))

    y_label = accident_classification(y_predict)
    train_time = time_is_normal[0] + time_prz_space_leak[0] + time_rcs_loca[0] + time_sg_2nd_side_leak[0] + time_sgtr[0]
    pre_time = time_is_normal[1] + time_prz_space_leak[1] + time_rcs_loca[1] + time_sg_2nd_side_leak[1] + time_sgtr[1]

    print('svm\t' + 'train_time\t' + str(train_time) + '\t' + 'predict_time\t' + str(pre_time))

    return y_label, [train_time, pre_time]


def adaboost_single_model(x_data_set, y_data_set, n_estimators, learning_rate) -> object:
    train_start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)
    adaboost = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME", n_estimators=n_estimators,
                                  learning_rate=learning_rate)
    adaboost.fit(x_train, y_train)
    train_end_time = time.time()

    train_time = train_end_time - train_start_time

    predict_start_time = time.time()
    y_predict = adaboost.predict(x_test_set)
    predict_end_time = time.time()

    predict_time = predict_end_time - predict_start_time

    return y_predict, [train_time, predict_time]


def adaboost_predict_model():
    y_data_set = np.hstack((np.full(5000, 1), np.zeros(20000)))

    x_data_set = np.vstack((x_data[0], x_data[1], x_data[2], x_data[3], x_data[4]))
    y_predict_is_normal, time_is_normal = adaboost_single_model(x_data_set, y_data_set, 1, 0.01)

    x_data_set = np.vstack((x_data[1], x_data[0], x_data[2], x_data[3], x_data[4]))
    y_predict_prz_space_leak, time_prz_space_leak = adaboost_single_model(x_data_set, y_data_set, 8, 1.13)

    x_data_set = np.vstack((x_data[2], x_data[0], x_data[1], x_data[3], x_data[4]))
    y_predict_rcs_loca, time_rcs_loca = adaboost_single_model(x_data_set, y_data_set, 1, 0.01)

    x_data_set = np.vstack((x_data[3], x_data[0], x_data[1], x_data[2], x_data[4]))
    y_predict_sg_2nd_side_leak, time_sg_2nd_side_leak = adaboost_single_model(x_data_set, y_data_set, 9, 1.13)

    x_data_set = np.vstack((x_data[4], x_data[0], x_data[1], x_data[2], x_data[3]))
    y_predict_sgtr, time_sgtr = adaboost_single_model(x_data_set, y_data_set, 8, 0.06)

    y_predict = np.vstack((y_predict_is_normal, y_predict_prz_space_leak, y_predict_rcs_loca,
                           y_predict_sg_2nd_side_leak, y_predict_sgtr))

    y_label = accident_classification(y_predict)
    train_time = time_is_normal[0] + time_prz_space_leak[0] + time_rcs_loca[0] + time_sg_2nd_side_leak[0] + time_sgtr[0]
    pre_time = time_is_normal[1] + time_prz_space_leak[1] + time_rcs_loca[1] + time_sg_2nd_side_leak[1] + time_sgtr[1]

    print('adaboost\t' + 'train_time\t' + str(train_time) + '\t' + 'predict_time\t' + str(pre_time))

    return y_label, [train_time, pre_time]


def gbdt_lr_single_model(x_data_set, y_data_set, n_estimators, learning_rate) -> object:
    train_start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)
    gbdt = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    gbdt.fit(x_train, y_train)
    gbdt_feature = gbdt.apply(x_train).reshape(-1, n_estimators)

    enc = OneHotEncoder()
    enc.fit(gbdt_feature)

    gbdt_best_feature = np.array(enc.transform(gbdt_feature).toarray())
    lr = LogisticRegression()
    lr.fit(gbdt_best_feature, y_train)
    train_end_time = time.time()

    train_time = train_end_time - train_start_time

    predict_start_time = time.time()
    test_gbdt_feature = gbdt.apply(x_test_set).reshape(-1, n_estimators)
    test_one_hot_feature = np.array(enc.transform(test_gbdt_feature).toarray())
    y_predict = lr.predict(test_one_hot_feature)
    predict_end_time = time.time()

    predict_time = predict_end_time - predict_start_time

    return y_predict, [train_time, predict_time]


def gbdt_lr_predict_model():
    y_data_set = np.hstack((np.full(5000, 1), np.zeros(20000)))

    x_data_set = np.vstack((x_data[0], x_data[1], x_data[2], x_data[3], x_data[4]))
    y_predict_is_normal, time_is_normal = gbdt_lr_single_model(x_data_set, y_data_set, 70, 0.0077)

    x_data_set = np.vstack((x_data[1], x_data[0], x_data[2], x_data[3], x_data[4]))
    y_predict_prz_space_leak, time_prz_space_leak = gbdt_lr_single_model(x_data_set, y_data_set, 90, 0.1668)

    x_data_set = np.vstack((x_data[2], x_data[0], x_data[1], x_data[3], x_data[4]))
    y_predict_rcs_loca, time_rcs_loca = gbdt_lr_single_model(x_data_set, y_data_set, 70, 0.0077)

    x_data_set = np.vstack((x_data[3], x_data[0], x_data[1], x_data[2], x_data[4]))
    y_predict_sg_2nd_side_leak, time_sg_2nd_side_leak = gbdt_lr_single_model(x_data_set, y_data_set, 80, 0.4641)

    x_data_set = np.vstack((x_data[4], x_data[0], x_data[1], x_data[2], x_data[3]))
    y_predict_sgtr, time_sgtr = gbdt_lr_single_model(x_data_set, y_data_set, 70, 0.0077)

    y_predict = np.vstack((y_predict_is_normal, y_predict_prz_space_leak, y_predict_rcs_loca,
                           y_predict_sg_2nd_side_leak, y_predict_sgtr))

    y_label = accident_classification(y_predict)
    train_time = time_is_normal[0] + time_prz_space_leak[0] + time_rcs_loca[0] + time_sg_2nd_side_leak[0] + time_sgtr[0]
    pre_time = time_is_normal[1] + time_prz_space_leak[1] + time_rcs_loca[1] + time_sg_2nd_side_leak[1] + time_sgtr[1]

    print('gbdt\t' + 'train_time\t' + str(train_time) + '\t' + 'predict_time\t' + str(pre_time))

    return y_label, [train_time, pre_time]


def xgboost_predict_model() -> object:
    x_data_set = np.vstack((x_data[0], x_data[1], x_data[2], x_data[3], x_data[4]))
    y_data_set = np.hstack((np.full(len(x_data[0]), 0), np.full(len(x_data[1]), 1), np.full(len(x_data[2]), 2),
                            np.full(len(x_data[3]), 3), np.full(len(x_data[4]), 4)))

    train_start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)

    xgboost_model = xgboost.XGBClassifier(learning_rate=0.3, n_estimators=1, objective='multi:softmax', random_state=0,
                                          silent=True)
    xgboost_model.fit(x_train, y_train)
    train_end_time = time.time()

    train_time = train_end_time - train_start_time

    predict_start_time = time.time()
    y_predict = xgboost_model.predict(x_test_set)
    predict_end_time = time.time()

    predict_time = predict_end_time - predict_start_time

    print('xgboost\t' + 'train_time\t' + str(train_time) + '\t' + 'predict_time\t' + str(predict_time))

    return y_predict, [train_time, predict_time]


def classification_accuracy(y_label, y_true) -> object:
    label_number = len(y_label)
    classification_right_number = [0, 0, 0, 0, 0]
    classification_wrong_number = [0, 0, 0, 0, 0]
    unknown_number = [0, 0, 0, 0, 0]
    number = [0, 0, 0, 0, 0]

    for i_value in np.arange(label_number):
        if y_true[i_value] == 0 and y_label[i_value] != 100 and y_label[i_value] == y_true[i_value]:
            classification_right_number[0] = classification_right_number[0] + 1
            number[0] = number[0] + 1
        elif y_true[i_value] == 0 and y_label[i_value] == 100:
            unknown_number[0] = unknown_number[0] + 1
            number[0] = number[0] + 1
        elif y_true[i_value] == 0 and y_label[i_value] != y_true[i_value]:
            classification_wrong_number[0] = classification_wrong_number[0] + 1
            number[0] = number[0] + 1
        elif y_true[i_value] == 1 and y_label[i_value] != 100 and y_label[i_value] == y_true[i_value]:
            classification_right_number[1] = classification_right_number[1] + 1
            number[1] = number[1] + 1
        elif y_true[i_value] == 1 and y_label[i_value] == 100:
            unknown_number[1] = unknown_number[1] + 1
            number[1] = number[1] + 1
        elif y_true[i_value] == 1 and y_label[i_value] != y_true[i_value]:
            classification_wrong_number[1] = classification_wrong_number[1] + 1
            number[1] = number[1] + 1
        elif y_true[i_value] == 2 and y_label[i_value] != 100 and y_label[i_value] == y_true[i_value]:
            classification_right_number[2] = classification_right_number[2] + 1
            number[2] = number[2] + 1
        elif y_true[i_value] == 2 and y_label[i_value] == 100:
            unknown_number[2] = unknown_number[2] + 1
            number[2] = number[2] + 1
        elif y_true[i_value] == 2 and y_label[i_value] != y_true[i_value]:
            classification_wrong_number[2] = classification_wrong_number[2] + 1
            number[2] = number[2] + 1
        elif y_true[i_value] == 3 and y_label[i_value] != 100 and y_label[i_value] == y_true[i_value]:
            classification_right_number[3] = classification_right_number[3] + 1
            number[3] = number[3] + 1
        elif y_true[i_value] == 3 and y_label[i_value] == 100:
            unknown_number[3] = unknown_number[3] + 1
            number[3] = number[3] + 1
        elif y_true[i_value] == 3 and y_label[i_value] != y_true[i_value]:
            classification_wrong_number[3] = classification_wrong_number[3] + 1
            number[3] = number[3] + 1
        elif y_true[i_value] == 4 and y_label[i_value] != 100 and y_label[i_value] == y_true[i_value]:
            classification_right_number[4] = classification_right_number[4] + 1
            number[4] = number[4] + 1
        elif y_true[i_value] == 4 and y_label[i_value] == 100:
            unknown_number[4] = unknown_number[4] + 1
            number[4] = number[4] + 1
        elif y_true[i_value] == 4 and y_label[i_value] != y_true[i_value]:
            classification_wrong_number[4] = classification_wrong_number[4] + 1
            number[4] = number[4] + 1

    accuracy = [classification_right_number[i_value] / number[i_value] for i_value in np.arange(5)]
    wrong = [classification_wrong_number[i_value] / number[i_value] for i_value in np.arange(5)]
    unknown = [unknown_number[i_value] / number[i_value] for i_value in np.arange(5)]

    accuracy_all = sum(classification_right_number) / sum(number)
    wrong_all = sum(classification_wrong_number) / sum(number)
    unknown_all = sum(unknown_number) / sum(number)

    return [classification_right_number, classification_wrong_number, unknown_number, accuracy, wrong, unknown,
            accuracy_all, wrong_all, unknown_all]


def find_best_classification(y_label) -> object:
    y_best_label = []
    length = len(y_label)

    for i_value in np.arange(length):
        temp_list = y_label[i_value]
        temp_dictionary = Counter(temp_list)
        temp_dictionary = dict(temp_dictionary)
        sort_dictionary = sorted(temp_dictionary.items(), key=lambda d: d[1], reverse=True)
        if sort_dictionary[0][1] > 2:
            y_best_label.append(sort_dictionary[0][0])
        else:
            y_best_label.append(100)

    return y_best_label
