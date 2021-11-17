from data_pre_processing.fault_diagnosis_data_5000 import power_decreasing, prz_space_leak, rcs_loca, \
    sg_2nd_side_leak, sgtr
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
import os
import matplotlib.pyplot as plt


father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'Heatmap\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_document = result_directory + 'heatmap.txt'
if os.path.exists(result_document):
    os.remove(result_document)

standard_scale = StandardScaler()

power_decreasing = power_decreasing[:5000, :24]
power_decreasing_scale = standard_scale.fit_transform(power_decreasing)
prz_space_leak = prz_space_leak[:5000, :24]
prz_space_leak_scale = standard_scale.fit_transform(prz_space_leak)
rcs_loca = rcs_loca[:5000, :24]
rcs_loca_scale = standard_scale.fit_transform(rcs_loca)
sg_2nd_side_leak = sg_2nd_side_leak[:5000, :24]
sg_2nd_side_leak_scale = standard_scale.fit_transform(sg_2nd_side_leak)
sgtr = sgtr[:5000, :24]
sgtr_scale = standard_scale.fit_transform(sgtr)

power_decreasing_cov = np.cov(np.transpose(power_decreasing_scale))
prz_space_leak_cov = np.cov(np.transpose(prz_space_leak_scale))
rcs_loca_cov = np.cov(np.transpose(rcs_loca_scale))
sg_2nd_side_leak_cov = np.cov(np.transpose(sg_2nd_side_leak_scale))
sgtr_cov = np.cov(np.transpose(sgtr_scale))

power_prz_dis = pairwise_distances(np.transpose(power_decreasing), np.transpose(prz_space_leak), metric='cosine')
power_rcs_dis = pairwise_distances(np.transpose(power_decreasing), np.transpose(rcs_loca), metric='cosine')
power_sg_dis = pairwise_distances(np.transpose(power_decreasing), np.transpose(sg_2nd_side_leak), metric='cosine')
power_sgtr_dis = pairwise_distances(np.transpose(power_decreasing), np.transpose(sgtr), metric='cosine')


def write_data(data, name):
    data_shape = np.shape(data)
    columns = data_shape[0]
    rows = data_shape[1]
    with open(result_document, 'a+') as f:
        f.write(name + '\n')
        for column in range(columns):
            for row in range(rows):
                if row != rows - 1:
                    f.write(str(data[column][row]) + ',')
                else:
                    f.write(str(data[column][row]) + '\n')


write_data(power_decreasing_cov, "power_decreasing_cov")
write_data(prz_space_leak_cov, "prz_space_leak_cov")
write_data(rcs_loca_cov, "rcs_loca_cov")
write_data(sg_2nd_side_leak_cov, "sg_2nd_side_leak_cov")
write_data(sgtr_cov, "sgtr_cov")

write_data(power_prz_dis, "power_prz_dis")
write_data(power_rcs_dis, "power_rcs_dis")
write_data(power_sg_dis, "power_sg_dis")
write_data(power_sgtr_dis, "power_sgtr_dis")


# def draw_picture_cov(cov, name):
#     sns.set()
#     plt.figure(figsize=[3, 2.25])
#     heatmap = sns.heatmap(cov, cmap='binary', cbar=False)
#     cb = heatmap.figure.colorbar(heatmap.collections[0])
#     cb.ax.tick_params(labelsize=9)
#     plt.xlabel('Data set labels', fontdict={'family': 'Times New Roman', 'size': 10.5})
#     plt.ylabel('Data set labels', fontdict={'family': 'Times New Roman', 'size': 10.5})
#     plt.xticks(fontproperties='Times New Roman', size=10.5)
#     plt.yticks(fontproperties='Times New Roman', size=10.5)
#     plt.savefig(result_directory + name + '_cov.png')
#
#
# draw_picture_cov(power_decreasing_cov, "power_decreasing_cov")
print('done...')
