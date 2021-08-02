from IO_function import *

clear_file("PRLL_pressurizer_water_temperature")
normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

index = 24
rows = 0
data_array = np.zeros((301, 10))

# while rows < 10:
#     i_index = rows * 301
#     for j_index in range(301):
#         data_array[j_index][rows] = normal_data[i_index][index]
#         i_index = i_index + 1
#     rows = rows + 1
#
# data_array.tolist()
# length = len(data_array)
# for i_index in range(length):
#     write_to_text("PRLL_steam_outlet_flow_rate", data_array[i_index], "a+")


while rows < 10:
    i_index = rows * 301
    for j_index in range(301):
        data_array[j_index][rows] = anomaly_data[i_index][index]
        i_index = i_index + 1
    rows = rows + 1

data_array.tolist()
length = len(data_array)
for i_index in range(length):
    write_to_text("PRLL_pressurizer_water_temperature", data_array[i_index], "a+")

print("done...")

