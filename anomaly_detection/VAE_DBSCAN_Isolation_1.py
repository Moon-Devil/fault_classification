from VAE_function import *
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def VAE_DBSCAN_Isolation_function(data, data_isolation, filename):
    clear_file(filename)
    data_length = len(data)

    for index in np.arange(data_length):
        record_list = [data[index][0], data[index][1], data_isolation[index]]
        write_to_text(filename, record_list, "a+")


def VAE_DBSCAN_Isolation_separate_function(data, data_isolation, filename1, filename2) -> int:
    clear_file(filename1)
    clear_file(filename2)
    data_length = len(data)
    count_ = 0

    for index in np.arange(data_length):
        if data_isolation[index] == 1:
            record_list = [data[index][0], data[index][1], data_isolation[index]]
            write_to_text(filename1, record_list, "a+")
        else:
            record_list = [data[index][0], data[index][1], data_isolation[index]]
            write_to_text(filename2, record_list, "a+")
            count_ += 1
    return count_


normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

vae = VAE(25)

vae.fit(normal_data[: 5000, ], 201)

normal = vae.predict(normal_data[: 1204, ], "point", "VAE_Isolation_normal_point")
PRLL = vae.predict(anomaly_data[:1204, 0: 25], "point", "VAE_Isolation_prz_liquid_predict_point")
PRSL = vae.predict(anomaly_data[:1204, 25: 50], "point", "VAE_Isolation_prz_vapour_predict_point")
RCS_CL = vae.predict(anomaly_data[:1204, 50: 75], "point", "VAE_Isolation_rcs_cl_predict_point")
RCS_HL = vae.predict(anomaly_data[:1204, 75: 100], "point", "VAE_Isolation_rcs_hl_predict_point")
SG2L = vae.predict(anomaly_data[:1204, 100: 125], "point", "VAE_Isolation_sg_2nd_predict_point")
SGTR = vae.predict(anomaly_data[:1204, 125: 150], "point", "VAE_Isolation_sgtr_predict_point")

normal = np.array(normal)
PRLL = np.array(PRLL)
PRSL = np.array(PRSL)
RCS_CL = np.array(RCS_CL)
RCS_HL = np.array(RCS_HL)
SG2L = np.array(SG2L)
SGTR = np.array(SGTR)

model_isolation = IsolationForest(n_estimators=5000)
history = model_isolation.fit(normal)

normal_isolation = model_isolation.predict(normal)
PRLL_isolation = model_isolation.predict(PRLL)
PRSL_isolation = model_isolation.predict(PRSL)
RCS_CL_isolation = model_isolation.predict(RCS_CL)
RCS_HL_isolation = model_isolation.predict(RCS_HL)
SG2L_isolation = model_isolation.predict(SG2L)
SGTR_isolation = model_isolation.predict(SGTR)

point = np.vstack((PRLL, PRSL, RCS_CL, RCS_HL, SG2L, SGTR))
isolation = np.hstack((PRLL_isolation, PRSL_isolation, RCS_CL_isolation, RCS_HL_isolation, SG2L_isolation,
                       SGTR_isolation))
color_1 = []
color_2 = []
length = len(isolation)
for i_index in np.arange(length):
    if isolation[i_index] == 1:
        color_1.append("#A9A9A9")
    else:
        color_1.append("#808080")

length = len(normal_isolation)
for i_index in np.arange(length):
    if isolation[i_index] == 1:
        color_2.append("#FF0000")
    else:
        color_2.append("#FFA500")

color_1 = np.array(color_1)
color_2 = np.array(color_2)
plt.scatter(point[:, 0], point[:, 1], c=color_1, s=10)
plt.scatter(normal[:, 0], normal[:, 1], c=color_2, s=10)
plt.show()

VAE_DBSCAN_Isolation_separate_function(normal, normal_isolation, "1_VAE_Isolation_normal", "1_VAE_Isolation_anomaly")
VAE_DBSCAN_Isolation_separate_function(point, isolation, "1_VAE_Isolation_right", "1_VAE_Isolation_wrong")

VAE_DBSCAN_Isolation_function(normal[:602, ], normal_isolation[:602, ], "2_VAE_Isolation_normal_all")
VAE_DBSCAN_Isolation_function(PRLL[:602, ], PRLL_isolation[:602, ], "2_VAE_Isolation_PRLL")
VAE_DBSCAN_Isolation_function(PRSL[:602, ], PRSL_isolation[:602, ], "2_VAE_Isolation_PRSL")
VAE_DBSCAN_Isolation_function(RCS_CL[:602, ], RCS_CL_isolation[:602, ], "2_VAE_Isolation_RCS_CL")
VAE_DBSCAN_Isolation_function(RCS_HL[:602, ], RCS_HL_isolation[:602, ], "2_VAE_Isolation_RCS_HL")
VAE_DBSCAN_Isolation_function(SG2L[:602, ], SG2L_isolation[:602, ], "2_VAE_Isolation_SG2L")
VAE_DBSCAN_Isolation_function(SGTR[:602, ], SGTR_isolation[:602, ], "2_VAE_Isolation_SGTR")

print("done...")
