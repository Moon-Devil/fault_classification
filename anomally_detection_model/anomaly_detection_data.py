from data_pre_processing.fault_diagnosis_data_5000 import power_decreasing, prz_space_leak, rcs_loca, \
    sg_2nd_side_leak, sgtr
import numpy as np


power_decreasing = power_decreasing[3000:4000, ]
prz_space_leak = prz_space_leak[3000:4000, ]
rcs_loca = rcs_loca[3000:4000, ]
sg_2nd_side_leak = sg_2nd_side_leak[3000:4000, ]
sgtr = sgtr[3000:4000, ]

x_data_set = np.vstack((power_decreasing, prz_space_leak, rcs_loca, sg_2nd_side_leak, sgtr))
x_data = [power_decreasing, prz_space_leak, rcs_loca, sg_2nd_side_leak, sgtr]
