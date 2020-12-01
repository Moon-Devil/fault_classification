import MySQLdb
from data_pre_processing.fault_diagnosis_data_function import *


# 连接数据库
db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
cursor = db.cursor()

# 读取prz_liquid_space_leak数据
sql = "USE prz_liquid_space_leak"
cursor.execute(sql)
prz_liquid_space_leak, prz_liquid_space_leak_data_length = read_data(cursor)

# 读取prz_vapour_space_leak数据
sql = "USE prz_vapour_space_leak"
cursor.execute(sql)
prz_vapour_space_leak, prz_vapour_space_leak_data_length = read_data(cursor)

# 读取RCS CL LOCA 1数据
sql = "USE rcs_cl_loca_1"
cursor.execute(sql)
rcs_cl_loca_1, rcs_cl_loca_1_data_length = read_data(cursor)

# 读取RCS CL LOCA 2数据
sql = "USE rcs_cl_loca_2"
cursor.execute(sql)
rcs_cl_loca_2, rcs_cl_loca_2_data_length = read_data(cursor)

# 读取RCS HL LOCA 1数据
sql = "USE rcs_hl_loca_1"
cursor.execute(sql)
rcs_hl_loca_1, rcs_hl_loca_1_data_length = read_data(cursor)

# 读取RCS HL LOCA 2数据
sql = "USE rcs_hl_loca_2"
cursor.execute(sql)
rcs_hl_loca_2, rcs_hl_loca_2_data_length = read_data(cursor)

# 读取SG 2nd side leak数据
sql = "USE sg_2nd_side_leak"
cursor.execute(sql)
sg_2nd_side_leak, sg_2nd_side_leak_data_length = read_data(cursor)

# 读取sgtr60 power数据
sql = "USE sgtr60_power"
cursor.execute(sql)
sgtr60_power, sgtr60_power_data_length = sgtr_read_data(cursor, 0.6)

# 读取sgtr power数据
sql = "USE sgtr_power"
cursor.execute(sql)
sgtr100_power, sgtr_power_data_length = sgtr_read_data(cursor, 1)

prz_space_leak = np.vstack((prz_liquid_space_leak, prz_vapour_space_leak))
rcs_loca = np.vstack((rcs_cl_loca_1, rcs_cl_loca_2, rcs_hl_loca_1, rcs_hl_loca_2))
sgtr = np.vstack((sgtr60_power, sgtr100_power))

# 读取power decreasing数据
sql = "USE power_decreasing"
cursor.execute(sql)
power_decreasing, power_decreasing_data_length = power_decreasing_read_data(cursor)
