import numpy as np

host = "localhost"  # 数据库地址
username = "root"   # 数据库用户名
passwd = "1111"     # 数据库密码
database = "mysql"  # 数据库类型


def read_data(cursor, data_set_length) -> object:
    sql = "show tables;"
    cursor.execute(sql)
    table_names = []
    results = cursor.fetchall()
    for result in results:
        table_names.append(result[0])

    data_length = []
    data = 0
    table_names_length = len(table_names)
    for index in np.arange(1, table_names_length):
        sql = "select * from " + table_names[index]
        cursor.execute(sql)
        results = cursor.fetchall()

        # 读数据
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
        if column_length != 0:
            data_length.append(column_length)
            row_length = len(data_list[0])
            data_array = np.zeros(shape=(column_length, row_length))
            for column in np.arange(column_length):
                for row in np.arange(row_length):
                    data_array[column][row] = data_list[column][row]

            data_array = np.delete(data_array, [0, 15, 16, 19, 20], axis=1)
            index_name = table_names[index].split('_')
            if str.isdigit(index_name[-1][0]):
                severity = "0." + index_name[-1]
            else:
                severity = "".join(filter(str.isdigit,  index_name[-1]))

            severity = float(severity)
            add_row = np.full(column_length, severity)
            data_array = np.column_stack((data_array, add_row))
            data_array = data_array[:data_set_length, ]

            if index == 1:
                data = data_array
            else:
                data = np.vstack((data, data_array))

    add_row = np.zeros(len(data))
    data = np.column_stack((data, add_row))

    return data, data_length


def sgtr_read_data(cursor, power_ratio, data_set_length) -> object:
    data, data_length = read_data(cursor, data_set_length)
    column = len(data)
    for i_value in np.arange(column):
        data[i_value][-1] = power_ratio

    return data, data_length


def power_decreasing_read_data(cursor, data_set_length) -> object:
    sql = "show tables;"
    cursor.execute(sql)
    table_names = []
    results = cursor.fetchall()
    for result in results:
        table_names.append(result[0])

    data_length = []
    data = 0
    table_names_length = len(table_names)
    for index in np.arange(4, table_names_length):
        sql = "select * from " + table_names[index]
        cursor.execute(sql)
        results = cursor.fetchall()

        # 读数据
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
        if column_length != 0:
            data_length.append(column_length)
            row_length = len(data_list[0])
            data_array = np.zeros(shape=(column_length, row_length))
            for column in np.arange(column_length):
                for row in np.arange(row_length):
                    data_array[column][row] = data_list[column][row]

            data_array = np.delete(data_array, [0, 15, 16, 19, 20], axis=1)
            index_name = table_names[index].split('_')
            index_first_name = "".join(filter(str.isdigit,  index_name[-2]))
            index_second_name = "".join(filter(str.isdigit, index_name[-1]))

            power = float(index_first_name)
            add_row_1 = np.full(column_length, power)
            rate = float(index_second_name)
            add_row_2 = np.full(column_length, rate)
            data_array = np.column_stack((data_array, add_row_1, add_row_2))
            data_array = data_array[:data_set_length, ]

            if index == 4:
                data = data_array
            else:
                data = np.vstack((data, data_array))

    return data, data_length
