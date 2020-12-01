import numpy as np
from sklearn.externals import joblib
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


def is_power_decreasing(data) -> object:
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    y_true = []
    y_predict = []

    for i_value in np.arange(data[0].shape[0]):
        flag = False
        if data[0][i_value] == 3:
            flag = True

        if flag:
            TP = TP + 1
            y_predict.append(1)
        else:
            FN = FN + 1
            y_predict.append(0)
        y_true.append(1)

    negative_sample = [data[1], data[2], data[3], data[4]]

    for k_value in np.arange(len(negative_sample)):
        for i_value in np.arange(negative_sample[k_value].shape[0]):
            flag = False
            if negative_sample[k_value][i_value] == 3:
                flag = True

            if flag:
                FP = FP + 1
                y_predict.append(1)
            else:
                TN = TN + 1
                y_predict.append(0)
            y_true.append(0)

    roc_curve = roc_curve_function(y_true, y_predict)

    return [TP, FN, FP, TN], roc_curve


def is_prz_space_leak(data) -> object:
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    y_true = []
    y_predict = []

    for i_value in np.arange(data[1].shape[0]):
        flag = False
        if data[1][i_value] == 0:
            flag = True

        if flag:
            TP = TP + 1
            y_predict.append(1)
        else:
            FN = FN + 1
            y_predict.append(0)
        y_true.append(1)

    negative_sample = [data[0], data[2], data[3], data[4]]

    for k_value in np.arange(len(negative_sample)):
        for i_value in np.arange(negative_sample[k_value].shape[0]):
            flag = False
            if negative_sample[k_value][i_value] == 0:
                flag = True

            if flag:
                FP = FP + 1
                y_predict.append(1)
            else:
                TN = TN + 1
                y_predict.append(0)
            y_true.append(0)

    roc_curve = roc_curve_function(y_true, y_predict)

    return [TP, FN, FP, TN], roc_curve


def is_rcs_loca(data) -> object:
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    y_true = []
    y_predict = []

    for i_value in np.arange(data[2].shape[0]):
        flag = False
        if data[2][i_value] == 0:
            flag = True

        if flag:
            TP = TP + 1
            y_predict.append(1)
        else:
            FN = FN + 1
            y_predict.append(0)
        y_true.append(1)

    negative_sample = [data[0], data[1], data[3], data[4]]

    for k_value in np.arange(len(negative_sample)):
        for i_value in np.arange(negative_sample[k_value].shape[0]):
            flag = False
            if negative_sample[k_value][i_value] == 0:
                flag = True

            if flag:
                FP = FP + 1
                y_predict.append(1)
            else:
                TN = TN + 1
                y_predict.append(0)
            y_true.append(0)

    roc_curve = roc_curve_function(y_true, y_predict)

    return [TP, FN, FP, TN], roc_curve


def is_sg_2nd_side_leak(data) -> object:
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    y_true = []
    y_predict = []

    for i_value in np.arange(data[3].shape[0]):
        flag = False
        if data[3][i_value] == 0:
            flag = True

        if flag:
            TP = TP + 1
            y_predict.append(1)
        else:
            FN = FN + 1
            y_predict.append(0)
        y_true.append(1)

    negative_sample = [data[0], data[1], data[2], data[4]]

    for k_value in np.arange(len(negative_sample)):
        for i_value in np.arange(negative_sample[k_value].shape[0]):
            flag = False
            if negative_sample[k_value][i_value] == 0:
                flag = True

            if flag:
                FP = FP + 1
                y_predict.append(1)
            else:
                TN = TN + 1
                y_predict.append(0)
            y_true.append(0)

    roc_curve = roc_curve_function(y_true, y_predict)

    return [TP, FN, FP, TN], roc_curve


def is_sgtr(data) -> object:
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    y_true = []
    y_predict = []

    for i_value in np.arange(data[4].shape[0]):
        flag = False
        if data[4][i_value] == 1:
            flag = True

        if flag:
            TP = TP + 1
            y_predict.append(1)
        else:
            FN = FN + 1
            y_predict.append(0)
        y_true.append(1)

    negative_sample = [data[0], data[1], data[2], data[3]]

    for k_value in np.arange(len(negative_sample)):
        for i_value in np.arange(negative_sample[k_value].shape[0]):
            flag = False
            if negative_sample[k_value][i_value] == 1:
                flag = True

            if flag:
                FP = FP + 1
                y_predict.append(1)
            else:
                TN = TN + 1
                y_predict.append(0)
            y_true.append(0)

    roc_curve = roc_curve_function(y_true, y_predict)

    return [TP, FN, FP, TN], roc_curve


def gmm_model_function(result_directory, x_data):
    model_path = result_directory + "gmm_model" + "\\gmm_model.m"
    gmm_model = joblib.load(model_path)

    y_predict_power_decreasing = gmm_model.predict(x_data[0])
    y_predict_prz_space_leak = gmm_model.predict(x_data[1])
    y_predict_rcs_loca = gmm_model.predict(x_data[2])
    y_predict_sg_2nd_side_leak = gmm_model.predict(x_data[3])
    y_predict_sgtr = gmm_model.predict(x_data[4])

    y_data = [y_predict_power_decreasing, y_predict_prz_space_leak, y_predict_rcs_loca, y_predict_sg_2nd_side_leak,
              y_predict_sgtr]

    is_power_decreasing_result, is_power_decreasing_roc = is_power_decreasing(y_data)
    is_prz_space_leak_result, is_prz_space_leak_roc = is_prz_space_leak(y_data)
    is_rcs_loca_result, is_rcs_loca_roc = is_rcs_loca(y_data)
    is_sg_2nd_side_leak_result, is_sg_2nd_side_leak_roc = is_sg_2nd_side_leak(y_data)
    is_sgtr_result, is_sgtr_roc = is_sgtr(y_data)

    result_path = result_directory + "gmm_roc_result.txt"
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
