from IO_function import *


clear_file("SGTR_origin_data")
normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

array = anomaly_data[:301, 125: 150]

max_list = []
min_list = []
scala = np.zeros((301, 25))

for i_index in np.arange(25):
    max_list.append(max(array[:, i_index]))
    min_list.append(min(array[:, i_index]))

for i_index in np.arange(301):
    for j_index in np.arange(25):
        scala[i_index][j_index] = (max_list[j_index] - array[i_index][j_index]) / (max_list[j_index] - min_list[j_index])

for i_index in np.arange(301):
    write_to_text("SGTR_origin_data", scala[i_index, ].tolist(), "a+")
