import numpy as np

from IO_function import *

normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values


def heatmap_function(input_data, file_name):
    input_data_T = np.transpose(input_data)
    input_data_heatmap = np.cov(input_data_T)
    input_data_abs = np.abs(input_data_heatmap)

    columns = input_data_abs.shape[0]
    rows = input_data_abs.shape[1]
    for column in np.arange(columns):
        for row in np.arange(rows):
            if input_data_abs[column][row] > 100:
                input_data_abs[column][row] = 100

    for column in np.arange(columns):
        for row in np.arange(rows):
            input_data_abs[column][row] = input_data_abs[column][row] / 100

    clear_file(file_name)
    input_data_abs.tolist()
    length = len(input_data_abs)
    for i_index in np.arange(length):
        write_to_text(file_name, input_data_abs[i_index], "a+")

    heatmap_mean = np.mean(np.array(input_data_abs), axis=0)
    write_to_text(file_name, heatmap_mean, "a+")


heatmap_function(normal_data[: 5000], "heatmap_normal")
heatmap_function(anomaly_data[:5000, 0: 25], "heatmap_prz_liquid")
heatmap_function(anomaly_data[:5000, 50: 75], "heatmap_rcs_cl")
heatmap_function(anomaly_data[:5000, 125: 150], "heatmap_sgtr")
