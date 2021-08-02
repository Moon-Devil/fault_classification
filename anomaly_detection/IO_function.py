import torch
import numpy as np
import MySQLdb
import pandas as pd
import os


def read_data_from_database(database_name, database_number, database_length, flag) -> object:
    host = "localhost"  # 数据库地址
    username = "root"  # 数据库用户名
    passwd = "1111"  # 数据库密码
    database = "mysql"  # 数据库类型

    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    sql = "USE " + database_name
    cursor.execute(sql)

    sql = "show tables;"
    cursor.execute(sql)
    table_names = []
    results = cursor.fetchall()
    for result in results:
        table_names.append(result[0])

    data = 0
    if flag == "normal":
        start = 4
        end = len(table_names) - 2
    else:
        start = 1
        end = database_number + 1

    for index in np.arange(start, end):
        sql = "select * from " + table_names[index]
        cursor.execute(sql)
        results = cursor.fetchall()

        data_list = []
        column_index = 0
        for row in results:
            temp = []
            row_length = len(row)
            for j_value in np.arange(row_length):
                temp.append(row[j_value])
            data_list.append(temp)
            column_index = column_index + 1

        column_length = len(data_list)
        if column_length != 0 and column_length > database_length:
            row_length = len(data_list[0])
            data_array = np.zeros(shape=(database_length, row_length))
            for column in np.arange(database_length):
                for row in np.arange(row_length):
                    data_array[column][row] = data_list[column][row]

            data_array = np.delete(data_array, [0, 15, 16, 19, 20], axis=1)

            if index == start:
                data = data_array
            else:
                data = np.vstack((data, data_array))
        else:
            print("The length of database is smaller than " + str(database_length) + ".")

    table_header = ['thermal_power', 'electric_power',
                    'coolant_flow_primary_circuit', 'coolant_flow_secondary_circuit',
                    'hot_leg_temperature_primary', 'hot_leg_temperature_secondary',
                    'cold_leg_temperature_primary', 'cold_leg_temperature_secondary',
                    'pressure_steam_generator_primary', 'pressure_steam_generator_secondary',
                    'water_level_primary', 'water_level_secondary',
                    'feed_water_flow_steam_generator_1', 'feed_water_flow_steam_generator_2',
                    'feed_water_temp_steam_generator_1', 'feed_water_temp_steam_generator_2',
                    'steam_outlet_flow_primary', 'steam_outlet_flow_secondary',
                    'steam_outlet_temperature_primary', 'steam_outlet_temperature_secondary',
                    'pressurizer_pressure', 'pressurizer_water_level',
                    'pressurizer_heat_power', 'pressurizer_steam_space_temperature',
                    'pressurizer_water_space_temperature']

    data_dataframe = pd.DataFrame(data, columns=table_header)
    return data_dataframe


def read_data() -> object:
    normal_data = read_data_from_database("power_decreasing", 0, 301, "normal")
    normal_data = normal_data.drop(normal_data.tail(20).index)

    prz_liquid = read_data_from_database("prz_liquid_space_leak", 20, 301, 0)

    prz_vapour = read_data_from_database("prz_vapour_space_leak", 20, 301, 0)

    rcs_cl_1 = read_data_from_database("rcs_cl_loca_1", 15, 201, 0)
    rcs_cl_2 = read_data_from_database("rcs_cl_loca_2", 15, 201, 0)
    rcs_cl = pd.concat([rcs_cl_1, rcs_cl_2], ignore_index=True)

    rcs_hl_1 = read_data_from_database("rcs_hl_loca_1", 15, 201, 0)
    rcs_hl_2 = read_data_from_database("rcs_hl_loca_2", 15, 201, 0)
    rcs_hl = pd.concat([rcs_hl_1, rcs_hl_2], ignore_index=True)

    sg_2nd = read_data_from_database("sg_2nd_side_leak", 20, 301, 0)

    sgtr60 = read_data_from_database("sgtr60_power", 15, 201, 0)
    sgtr100 = read_data_from_database("sgtr_power", 15, 201, 0)
    sgtr = pd.concat([sgtr60, sgtr100], ignore_index=True)

    anomaly_data_pandas = pd.concat([prz_liquid, prz_vapour, rcs_cl, rcs_hl, sg_2nd, sgtr], axis=1)

    table_header = ['thermal_power', 'electric_power',
                    'coolant_flow_primary_circuit', 'coolant_flow_secondary_circuit',
                    'hot_leg_temperature_primary', 'hot_leg_temperature_secondary',
                    'cold_leg_temperature_primary', 'cold_leg_temperature_secondary',
                    'pressure_steam_generator_primary', 'pressure_steam_generator_secondary',
                    'water_level_primary', 'water_level_secondary',
                    'feed_water_flow_steam_generator_1', 'feed_water_flow_steam_generator_2',
                    'feed_water_temp_steam_generator_1', 'feed_water_temp_steam_generator_2',
                    'steam_outlet_flow_primary', 'steam_outlet_flow_secondary',
                    'steam_outlet_temperature_primary', 'steam_outlet_temperature_secondary',
                    'pressurizer_pressure', 'pressurizer_water_level',
                    'pressurizer_heat_power', 'pressurizer_steam_space_temperature',
                    'pressurizer_water_space_temperature']

    first_index = [index for index in ["prz_liquid", "prz_vapour", "rcs_cl", "rcs_hl", "sg_2nd", "sgtr"]
                   for _ in np.arange(25)]
    second_index = [table_header[index] for _ in np.arange(6) for index in np.arange(len(table_header))]
    index = [first_index, second_index]

    anomaly_data = pd.DataFrame(anomaly_data_pandas.values, columns=index)
    anomaly_data = anomaly_data.drop(anomaly_data.tail(30).index)

    return normal_data, anomaly_data


def data_slice_function(data_set) -> object:
    length = data_set.shape[0]
    i_index = 0
    while i_index < length:
        data_batch = data_set[i_index,]
        i_index = i_index + 1
        yield data_batch


def write_to_text(filename, data_lists, flag):
    father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
    result_directory = father_path + 'anomaly_detection_paper\\'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    file = os.path.join(result_directory, filename + ".txt")
    with open(file, flag) as f:
        length = len(data_lists)
        for i_index in np.arange(length):
            if i_index != length - 1:
                f.write(str(data_lists[i_index]) + ",")
            else:
                f.write(str(data_lists[i_index]) + "\n")


def clear_file(filename):
    if filename is not None:
        father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
        result_directory = father_path + 'anomaly_detection_paper\\'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        file = os.path.join(result_directory, filename + ".txt")
        if os.path.exists(file):
            os.remove(file)


def write_parameter_to_text(data_lists, file_name):
    father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
    result_directory = father_path + 'anomaly_detection_paper\\'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    weights = data_lists[0].detach().numpy()
    h_bias = data_lists[1].detach().numpy()
    v_bias = data_lists[2].detach().numpy()
    persistent = data_lists[3].detach().numpy()

    file = os.path.join(result_directory, file_name + "_parameter.txt")
    with open(file, "w+") as f:
        columns = weights.shape[0]
        rows = weights.shape[1]
        f.write(str(columns) + "," + str(rows) + "\n")
        for column in np.arange(columns):
            for row in np.arange(rows):
                if row != rows - 1:
                    f.write(str(weights[column][row]) + ",")
                else:
                    f.write(str(weights[column][row]) + "\n")

        rows = h_bias.shape[0]
        f.write(str(rows) + "\n")
        for row in np.arange(rows):
            if row != rows - 1:
                f.write(str(h_bias[row]) + ",")
            else:
                f.write(str(h_bias[row]) + "\n")

        rows = v_bias.shape[0]
        f.write(str(rows) + "\n")
        for row in np.arange(rows):
            if row != rows - 1:
                f.write(str(v_bias[row]) + ",")
            else:
                f.write(str(v_bias[row]) + "\n")

        persistent = persistent.squeeze()
        rows = persistent.shape[0]
        f.write(str(rows) + "\n")
        for row in np.arange(rows):
            if row != rows - 1:
                f.write(str(persistent[row]) + ",")
            else:
                f.write(str(persistent[row]) + "\n")


def read_parameters_function(file_name) -> object:
    father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
    result_directory = father_path + 'anomaly_detection_paper\\'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    file = os.path.join(result_directory, file_name + "_parameters.txt")

    with open(file, "r") as f:
        data = f.read().split("\n")

        weights_line = data[0].split(",")
        weights_columns = int(weights_line[0])
        rows = int(weights_line[1])
        weights = np.zeros((weights_columns, rows))
        for column in np.arange(weights_columns):
            line = data[column + 1].split(",")
            for row in np.arange(rows):
                weights[column][row] = float(line[row])
        line_columns = weights_columns + 1

        h_bias_line = data[line_columns].split(",")
        rows = int(h_bias_line[0])
        h_bias = np.zeros(rows)
        line_columns = line_columns + 1
        line = data[line_columns].split(",")
        for row in np.arange(rows):
            h_bias[row] = float(line[row])
        line_columns = line_columns + 1

        v_bias_line = data[line_columns].split(",")
        rows = int(v_bias_line[0])
        v_bias = np.zeros(rows)
        line_columns = line_columns + 1
        line = data[line_columns].split(",")
        for row in np.arange(rows):
            v_bias[row] = float(line[row])
        line_columns = line_columns + 1

        parameters_line = data[line_columns].split(",")
        rows = int(parameters_line[0])
        parameters = np.zeros((1, rows))
        line_columns = line_columns + 1
        line = data[line_columns].split(",")
        for row in np.arange(rows):
            parameters[0][row] = float(line[row])

        weights = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
        h_bias = torch.tensor(h_bias, dtype=torch.float64, requires_grad=True)
        v_bias = torch.tensor(v_bias, dtype=torch.float64, requires_grad=True)
        parameters = torch.tensor(parameters, dtype=torch.float64)
    return weights, h_bias, v_bias, parameters
