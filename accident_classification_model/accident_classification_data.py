from data_pre_processing.fault_diagnosis_data_5000 import power_decreasing, prz_space_leak, rcs_loca, \
    sg_2nd_side_leak, sgtr
import numpy as np


row_0 = np.zeros(5000)
row_1 = np.full(5000, 1)

power_decreasing = power_decreasing[:5000, ]
power_decreasing_y = np.column_stack((row_0, row_0, row_0, row_0))
power_decreasing_y_1 = np.column_stack((row_0, row_0, row_0, row_0))

prz_space_leak = prz_space_leak[:5000, ]
prz_space_leak_y = np.column_stack((row_0, row_0, row_0, row_1))

rcs_loca = rcs_loca[:5000, ]
rcs_loca_y = np.column_stack((row_0, row_0, row_1, row_0))

sg_2nd_side_leak = sg_2nd_side_leak[:5000, ]
sg_2nd_side_leak_y = np.column_stack((row_0, row_0, row_1, row_1))
sg_2nd_side_leak_y_1 = np.column_stack((row_0, row_1, row_0, row_0))

sgtr = sgtr[:5000, ]
sgtr_y = np.column_stack((row_0, row_1, row_0, row_0))
sgtr_y_1 = np.column_stack((row_1, row_0, row_0, row_0))

x_data_set = np.vstack((power_decreasing, prz_space_leak, rcs_loca, sg_2nd_side_leak, sgtr))
y_data_set = np.vstack((power_decreasing_y, prz_space_leak_y, rcs_loca_y, sg_2nd_side_leak_y, sgtr_y))
y_data_set_1 = np.vstack((power_decreasing_y_1, prz_space_leak_y, rcs_loca_y, sg_2nd_side_leak_y_1, sgtr_y_1))

x_data = [power_decreasing, prz_space_leak, rcs_loca, sg_2nd_side_leak, sgtr]

