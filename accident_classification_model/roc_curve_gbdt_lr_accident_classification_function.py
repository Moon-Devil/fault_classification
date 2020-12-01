import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import os


def roc_curve_function(y_true, y_predict) -> object:
    roc_list = list(zip(y_predict, y_true))
    roc_curve = []
    x_coordinate = 0.0
    y_coordinate = 0.0
    number_positive = 0
    number_negative = 0
    length = len(roc_list)

    for i_value in np.arange(length):
        if roc_list[i_value][0] == 1 and roc_list[i_value][1] == 1:
            number_positive = number_positive + 1

        if roc_list[i_value][0] == 1 and roc_list[i_value][1] == 0:
            number_negative = number_negative + 1

    if number_negative == 0:
        roc_curve = [[0.0, 1.0], [1.0, 1.0]]
    else:
        for i_value in np.arange(length):
            if roc_list[i_value][0] == 1 and roc_list[i_value][1] == 1:
                y_coordinate = y_coordinate + 1.0 / number_positive
                roc_curve.append([x_coordinate, y_coordinate])

            if roc_list[i_value][0] == 1 and roc_list[i_value][1] == 0:
                x_coordinate = x_coordinate + 1.0 / number_negative
                roc_curve.append([x_coordinate, y_coordinate])

    return roc_curve


def gbdt_lr_classification(y_predict, y_true) -> object:
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    length = len(y_predict)
    for i_value in np.arange(length):
        if y_predict[i_value] == 1 and y_true[i_value] == 1:
            TP = TP + 1
        elif y_predict[i_value] == 0 and y_true[i_value] == 1:
            FN = FN + 1
        elif y_predict[i_value] == 1 and y_true[i_value] == 0:
            FP = FP + 1
        elif y_predict[i_value] == 0 and y_true[i_value] == 0:
            TN = TN + 1
        else:
            exit()

    roc_curve = roc_curve_function(y_true, y_predict)

    return [TP, FN, FP, TN], roc_curve


def gbdt_lr_model(x_data_set, y_data_set, n_estimators, learning_rate) -> object:
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.3)
    gbdt = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    gbdt.fit(x_train, y_train)
    gbdt_feature = gbdt.apply(x_train).reshape(-1, n_estimators)

    enc = OneHotEncoder()
    enc.fit(gbdt_feature)

    gbdt_best_feature = np.array(enc.transform(gbdt_feature).toarray())
    lr = LogisticRegression()
    lr.fit(gbdt_best_feature, y_train)

    test_gbdt_feature = gbdt.apply(x_test).reshape(-1, n_estimators)
    test_one_hot_feature = np.array(enc.transform(test_gbdt_feature).toarray())
    y_predict = lr.predict(test_one_hot_feature)

    return y_predict, y_test


def gbdt_lr_model_function(result_directory, x_data):
    y_data_set = np.hstack((np.full(5000, 1), np.zeros(20000)))

    x_data_set = np.vstack((x_data[0], x_data[1], x_data[2], x_data[3], x_data[4]))
    y_predict, y_true = gbdt_lr_model(x_data_set, y_data_set, 50, 0.01)
    is_power_decreasing_result, is_power_decreasing_roc = gbdt_lr_classification(y_predict, y_true)

    x_data_set = np.vstack((x_data[1], x_data[0], x_data[2], x_data[3], x_data[4]))
    y_predict, y_true = gbdt_lr_model(x_data_set,  y_data_set, 90, 0.1)
    is_prz_space_leak_result, is_prz_space_leak_roc = gbdt_lr_classification(y_predict, y_true)

    x_data_set = np.vstack((x_data[2], x_data[0], x_data[1], x_data[3], x_data[4]))
    y_predict, y_true = gbdt_lr_model(x_data_set,  y_data_set, 50, 0.01)
    is_rcs_loca_result, is_rcs_loca_roc = gbdt_lr_classification(y_predict, y_true)

    x_data_set = np.vstack((x_data[3], x_data[0], x_data[1], x_data[2], x_data[4]))
    y_predict, y_true = gbdt_lr_model(x_data_set,  y_data_set, 90, 0.1)
    is_sg_2nd_side_leak_result, is_sg_2nd_side_leak_roc = gbdt_lr_classification(y_predict, y_true)

    x_data_set = np.vstack((x_data[4], x_data[0], x_data[1], x_data[2], x_data[3]))
    y_predict, y_true = gbdt_lr_model(x_data_set, y_data_set, 50, 0.01)
    is_sgtr_result, is_sgtr_roc = gbdt_lr_classification(y_predict, y_true)

    result_path = result_directory + "gbdt_lr_roc_result.txt"
    if os.path.exists(result_path):
        os.remove(result_path)

    with open(result_path, "w+") as f:
        f.write("is_power_decreasing\t" + "TP\t" + str(is_power_decreasing_result[0]) + "\t" +
                "FN\t" + str(is_power_decreasing_result[1]) + "\t" + "FP\t" + str(is_power_decreasing_result[2]) + "\t"
                + "TN\t" + str(is_power_decreasing_result[3]) + "\n")
        length = len(is_power_decreasing_roc)
        for i_value in np.arange(length):
            f.write(str(is_power_decreasing_roc[i_value][0]) + "\t" + str(is_power_decreasing_roc[i_value][1]) + "\n")

        f.write("is_prz_space_leak\t" + "TP\t" + str(is_prz_space_leak_result[0]) + "\t" +
                "FN\t" + str(is_prz_space_leak_result[1]) + "\t" + "FP\t" + str(is_prz_space_leak_result[2]) + "\t"
                + "TN\t" + str(is_prz_space_leak_result[3]) + "\n")
        length = len(is_prz_space_leak_roc)
        for i_value in np.arange(length):
            f.write(str(is_prz_space_leak_roc[i_value][0]) + "\t" + str(is_prz_space_leak_roc[i_value][1]) + "\n")

        f.write("is_rcs_loca\t" + "TP\t" + str(is_rcs_loca_result[0]) + "\t" +
                "FN\t" + str(is_rcs_loca_result[1]) + "\t" + "FP\t" + str(is_rcs_loca_result[2]) + "\t"
                + "TN\t" + str(is_rcs_loca_result[3]) + "\n")
        length = len(is_rcs_loca_roc)
        for i_value in np.arange(length):
            f.write(str(is_rcs_loca_roc[i_value][0]) + "\t" + str(is_rcs_loca_roc[i_value][1]) + "\n")

        f.write("is_sg_2nd_side_leak\t" + "TP\t" + str(is_sg_2nd_side_leak_result[0]) + "\t" +
                "FN\t" + str(is_sg_2nd_side_leak_result[1]) + "\t" + "FP\t" + str(is_sg_2nd_side_leak_result[2]) + "\t"
                + "TN\t" + str(is_sg_2nd_side_leak_result[3]) + "\n")
        length = len(is_sg_2nd_side_leak_roc)
        for i_value in np.arange(length):
            f.write(str(is_sg_2nd_side_leak_roc[i_value][0]) + "\t" + str(is_sg_2nd_side_leak_roc[i_value][1]) + "\n")

        f.write("is_sgtr\t" + "TP\t" + str(is_sgtr_result[0]) + "\t" +
                "FN\t" + str(is_sgtr_result[1]) + "\t" + "FP\t" + str(is_sgtr_result[2]) + "\t"
                + "TN\t" + str(is_sgtr_result[3]) + "\n")
        length = len(is_sgtr_roc)
        for i_value in np.arange(length):
            f.write(str(is_sgtr_roc[i_value][0]) + "\t" + str(is_sgtr_roc[i_value][1]) + "\n")
