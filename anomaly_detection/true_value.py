from IO_function import *

normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

normal_data.tolist()

clear_file("True_value")
for i_index in np.arange(301):
    write_to_text("True_value", normal_data[i_index], "a+")
