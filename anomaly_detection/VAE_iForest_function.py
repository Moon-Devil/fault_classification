from VAE_function import *
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def VAE_iForest_write2text(data, filename):
    clear_file(filename)
    data_length = len(data)

    for index in np.arange(data_length):
        record_list = [data[index][0], data[index][1]]
        write_to_text(filename, record_list, "a+")


def VAE_iForest_predict(data, data_isolation, filename):
    clear_file(filename)
    data_length = len(data)

    for index in np.arange(data_length):
        record_list = [data[index][0], data[index][1], data_isolation[index]]
        write_to_text(filename, record_list, "a+")


def obtain_array(train_label, normal_data, anomaly_data, array_length):
    if train_label == "PowerR":
        return normal_data[: array_length, ]
    elif train_label == "PRLL_accident":
        return anomaly_data[: array_length, 0: 25]
    elif train_label == "PRSL_accident":
        return anomaly_data[: array_length, 25: 50]
    elif train_label == "CL_LOCA_accident":
        return anomaly_data[: array_length, 50: 75]
    elif train_label == "HL_LOCA_accident":
        return anomaly_data[: array_length, 75: 100]
    elif train_label == "SG2L_accident":
        return anomaly_data[: array_length, 100: 125]
    else:
        return anomaly_data[: array_length, 125: 150]


def VAE_iForest_train_function(vae, train_label, normal_data, anomaly_data):
    labels = ["PowerR", "PRLL_accident", "PRSL_accident", "CL_LOCA_accident", "HL_LOCA_accident", "SG2L_accident",
              "SGTR_accident"]
    labels.remove(train_label)

    normal_filename = "VAE_iForest_1_" + train_label + "_normal"
    outlier_filename = "VAE_iForest_1_" + train_label + "_outlier"
    right_filename = "VAE_iForest_1_" + train_label + "_right"
    wrong_filename = "VAE_iForest_1_" + train_label + "_wrong"

    train_data = obtain_array(train_label, normal_data, anomaly_data, 5000)
    vae.fit(train_data, 201)
    train_encoder = vae.predict(train_data, "point", None)
    train_encoder = np.array(train_encoder)

    print("Start train iForest model...")
    model_iForest = IsolationForest(n_estimators=5000)
    model_iForest.fit(train_encoder[: 1204, ])
    print("iForest model has been complied.")

    temp_data = obtain_array(train_label, normal_data, anomaly_data, 1204)
    temp_encoder = vae.predict(temp_data, "point", None)
    temp_encoder = np.array(temp_encoder)
    normal_iForest = model_iForest.predict(temp_encoder)

    text_data = obtain_array(labels[0], normal_data, anomaly_data, 1204)
    text_encoder = vae.predict(text_data, "point", None)
    text_encoder = np.array(text_encoder)
    text_iForest = model_iForest.predict(text_encoder)

    length = len(labels)
    for i_index in np.arange(1, length):
        temp_data = obtain_array(labels[i_index], normal_data, anomaly_data, 1204)
        temp_encoder = vae.predict(temp_data, "point", None)
        temp_encoder = np.array(temp_encoder)
        temp_iForest = model_iForest.predict(temp_encoder)

        text_encoder = np.vstack((text_encoder, temp_encoder))
        text_iForest = np.hstack((text_iForest, temp_iForest))

    normal = []
    outlier = []
    normal_count = 0
    outlier_count = 0
    length = len(normal_iForest)
    for i_index in np.arange(length):
        if normal_iForest[i_index] == 1:
            normal.append(train_encoder[i_index, ])
            normal_count += 1
        else:
            outlier.append(train_encoder[i_index, ])
            outlier_count += 1

    right = []
    wrong = []
    right_count = 0
    wrong_count = 0
    length = len(text_iForest)
    for i_index in np.arange(length):
        if text_iForest[i_index] == -1:
            right.append(text_encoder[i_index, ])
            right_count += 1
        else:
            wrong.append(text_encoder[i_index, ])
            wrong_count += 1

    normal = np.array(normal)
    outlier = np.array(outlier)
    right = np.array(right)
    wrong = np.array(wrong)

    VAE_iForest_write2text(normal, normal_filename)
    VAE_iForest_write2text(outlier, outlier_filename)
    VAE_iForest_write2text(right, right_filename)
    VAE_iForest_write2text(wrong, wrong_filename)

    plt.scatter(right[:, 0], right[:, 1], s=5)
    plt.scatter(wrong[:, 0], wrong[:, 1], s=5)
    plt.scatter(outlier[:, 0], outlier[:, 1], s=5)
    plt.scatter(normal[:, 0], normal[:, 1], s=5)
    plt.show()

    file_name_1 = "VAE_iForest_2_" + train_label + "_origin_data"
    file_name_2 = "VAE_iForest_2_" + train_label + "_reconstruction_data"
    clear_file(file_name_1)
    clear_file(file_name_2)
    origin_data = obtain_array(train_label, normal_data, anomaly_data, 601)
    reconstruction_data = vae.reconstruction_function(origin_data)

    for i_index in np.arange(601):
        record_list = origin_data[i_index, ].tolist()
        write_to_text(file_name_1, record_list, "a+")
        record_list = reconstruction_data[i_index, ].tolist()
        write_to_text(file_name_2, record_list, "a+")

    labels = ["PowerR", "PRLL_accident", "PRSL_accident", "CL_LOCA_accident", "HL_LOCA_accident", "SG2L_accident",
              "SGTR_accident"]

    length = len(labels)
    for i_index in np.arange(length):
        label = labels[i_index]
        file_name = "VAE_iForest_3_" + label + "_data_scala"
        clear_file(file_name)
        origin_data = obtain_array(label, normal_data, anomaly_data, 601)
        origin_data = np.transpose(origin_data)
        columns = np.shape(origin_data)[0]
        rows = np.shape(origin_data)[1]
        for j_index in np.arange(columns):
            temp_list = origin_data[j_index, ].tolist()
            max_value = max(temp_list)
            min_value = min(temp_list)
            record_list = []
            for k_index in np.arange(rows):
                value = (origin_data[j_index][k_index] - min_value) / (max_value - min_value)
                record_list.append(value)
            write_to_text(file_name, record_list, "a+")

    length = len(labels)
    for i_index in np.arange(length):
        label = labels[i_index]
        file_name = "VAE_iForest_4_" + label + "_situation"
        clear_file(file_name)
        temp_data = obtain_array(label, normal_data, anomaly_data, 601)
        start_predict_time = time.time()
        temp_encoder = vae.predict(temp_data, "point", None)
        temp_encoder = np.array(temp_encoder)
        temp_iForest = model_iForest.predict(temp_encoder)
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time
        write_to_text(file_name, temp_iForest, "a+")
        write_to_text(file_name, [predict_time], "a+")



    print("done....")



