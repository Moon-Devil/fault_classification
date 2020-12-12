from anomally_detection_model.anomaly_detection_data_1000 import x_data
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import os


father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'anomaly_detection\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

node_visible = 27
node_hidden = 500
learning_rate = 0.01
training_epochs = 20
inner_epochs = 5


class RBM(object):
    def __init__(self, input_value=x_data, index=0, n_visible=27, n_hidden=500, weight=None, h_bias=None, v_bias=None):
        self.scale_model = MinMaxScaler()
        self.index = index
        self.n_visible = tf.constant(n_visible)
        self.n_hidden = tf.constant(n_hidden)
        self.prob = tf.constant(1)
        self.input_value = input_value
        length = len(input_value)
        self.input_value_scale = []
        self.input_value_tensor = []
        for i_value in np.arange(length):
            self.input_value_scale.append(self.scale_model.fit_transform(input_value[i_value]))

        for i_value in np.arange(length):
            self.input_value_tensor.append(tf.cast(tf.constant(self.input_value_scale[i_value]), dtype=tf.float32))

        if weight is None:
            bounds = -4.0 * np.sqrt(6.0 / (n_visible + n_hidden))
            weight = tf.random_uniform([n_visible, n_hidden], minval=-bounds, maxval=bounds)
        if h_bias is None:
            h_bias = tf.zeros((self.n_hidden,))
        if v_bias is None:
            v_bias = tf.zeros((self.n_visible,))
        self.weight = tf.convert_to_tensor(weight)
        self.h_bias = tf.convert_to_tensor(h_bias)
        self.v_bias = tf.convert_to_tensor(v_bias)
        self.params = [self.weight, self.h_bias, self.v_bias]

    def energy_function(self, v_sample) -> object:
        h_prediction = tf.matmul(v_sample, self.weight) + self.h_bias
        v_bias_term = tf.matmul(v_sample, tf.expand_dims(self.v_bias, axis=1))
        hidden_term = tf.reduce_sum(tf.log(1.0 + tf.exp(h_prediction)), axis=1)

        return -hidden_term - v_bias_term

    def gibbs_hvh(self, h_sample) -> object:
        v1_value = tf.keras.activations.sigmoid(tf.matmul(h_sample, tf.transpose(self.weight)) + self.v_bias)
        v1_sample = tf.keras.activations.relu(tf.sign(v1_value - tf.random_uniform(tf.shape(v1_value))))

        h1_value = tf.keras.activations.sigmoid(tf.matmul(v1_sample, self.weight) + self.h_bias)
        h1_sample = tf.keras.activations.relu(tf.sign(h1_value - tf.random_uniform(tf.shape(h1_value))))

        return v1_value, v1_sample, h1_value, h1_sample

    @staticmethod
    def condition(i_value, in_epochs, nv_value, nv_sample, nh_value, nh_sample) -> object:
        return i_value < in_epochs

    def body(self, i_value, in_epochs, nv_value, nv_sample, nh_value, nh_sample) -> object:
        i_value = i_value + 1
        nv_value, nv_sample, nh_value, nh_sample = self.gibbs_hvh(nh_sample)
        return i_value, in_epochs, nv_value, nv_sample, nh_value, nh_sample

    def train_step(self, lr=0.01, in_epochs=1, persistent=None) -> object:
        h1_value = tf.keras.activations.sigmoid(tf.matmul(self.input_value_tensor[self.index], self.weight) +
                                                self.h_bias)
        h1_sample = tf.keras.activations.relu(tf.sign(h1_value - tf.random_uniform(tf.shape(h1_value))))
        if persistent is None:
            chain_start = h1_sample
        else:
            chain_start = persistent

        i_value, in_epochs, nv_value, nv_sample, nh_value, nh_sample = \
            tf.while_loop(self.condition, self.body,
                          loop_vars=[tf.constant(0), tf.constant(in_epochs),
                                     tf.zeros(tf.shape(self.input_value_tensor[self.index])),
                                     tf.zeros(tf.shape(self.input_value_tensor[self.index])),
                                     tf.zeros(tf.shape(chain_start)), chain_start])

        update_weight = self.weight + lr * (tf.matmul(tf.transpose(self.input_value_tensor[self.index]), h1_value) -
                                            tf.matmul(tf.transpose(nv_sample), nh_value)) / tf.to_float(
            tf.shape(self.input_value_tensor[self.index])[0])
        update_v_bias = self.v_bias + lr * (tf.reduce_mean(self.input_value_tensor[self.index] - nv_sample, axis=0))
        update_h_bias = self.h_bias + lr * (tf.reduce_mean(h1_value - nh_value, axis=0))

        self.weight = update_weight
        self.v_bias = update_v_bias
        self.h_bias = update_h_bias

        chain_end = tf.stop_gradient(nv_sample)
        cost = tf.reduce_mean(self.energy_function(self.input_value_tensor[self.index])) - tf.reduce_mean(
            self.energy_function(chain_end))
        g_param_m = tf.gradients(ys=[cost], xs=self.params)

        g_params = []
        for i_value in range(len(g_param_m)):
            g_params.append(tf.clip_by_value(g_param_m[i_value], clip_value_min=-1, clip_value_max=1))

        new_params = []
        for g_param, param in zip(g_params, self.params):
            param = param - g_param * lr
            new_params.append(param)

        if persistent is not None:
            persistent = nh_sample
            new_persistent = [persistent]
        else:
            new_persistent = []

        return new_params + new_persistent

    def loss_function(self) -> object:
        activation_h = tf.keras.activations.sigmoid(tf.matmul(self.input_value_tensor[self.index], self.weight) +
                                                    self.h_bias)
        activation_v = tf.keras.activations.sigmoid(tf.matmul(activation_h, tf.transpose(self.weight)) + self.v_bias)
        mean_squared_error = tf.sqrt(tf.reduce_mean(tf.square(self.input_value_tensor[self.index] - activation_v)))

        return mean_squared_error, activation_h, activation_v

    def predict(self, index) -> object:
        activation_h = tf.keras.activations.sigmoid(tf.matmul(self.input_value_tensor[index], self.weight) +
                                                    self.h_bias)
        activation_v = tf.keras.activations.sigmoid(tf.matmul(activation_h, tf.transpose(self.weight)) + self.v_bias)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            activation_h = sess.run(activation_h)
            activation_v = sess.run(activation_v)

        activation_v = self.scale_model.inverse_transform(activation_v)
        mse = np.sqrt(np.mean(np.square(self.input_value[index] - activation_v), axis=1))

        return mse, activation_h

    def fit(self, lr=learning_rate, t_epochs=training_epochs, i_epochs=inner_epochs):
        loss = []
        cost_time = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(t_epochs):
                epoch_start_time = time.time()
                self.train_step(lr=lr, in_epochs=i_epochs)
                loss_tensor, _, _ = self.loss_function()
                epoch_loss = sess.run(loss_tensor)
                loss.append(epoch_loss)
                epoch_end_time = time.time()
                epoch_cost_time = epoch_end_time - epoch_start_time
                cost_time.append(epoch_cost_time)
                log = "epoch={0}, loss={1}, time={2}"
                print(log.format(epoch, epoch_loss, epoch_cost_time))

        result_document = result_directory + 'train_epoch_loss.txt'
        if os.path.exists(result_document):
            os.remove(result_document)

        with open(result_document, 'w+') as f:
            f.write('epoch\t' + 'loss\t' + 'time\n')
            for i_value in np.arange(t_epochs):
                f.write(str(i_value) + '\t' + str(loss[i_value]) + '\t' + str(cost_time[i_value]) + '\n')


def DBN_train(x_train):
    start_train_time = time.time()
    rbm = RBM(x_train, 0)
    rbm.fit()

    mse, _ = rbm.predict(0)
    label = np.full(mse.shape[0], 100)
    label = np.hstack((label, np.full(mse.shape[0] * 4, -100)))

    for i_value in np.arange(1, 5, 1):
        mse_temp, _ = rbm.predict(i_value)
        mse = np.hstack((mse, mse_temp))
    mse = mse[:, np.newaxis]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, input_shape=[mse.shape[1], ]))
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    history = model.fit(mse, label, batch_size=32, epochs=200, validation_split=0.2)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    result_document = result_directory + 'DBN_train_result.txt'
    if os.path.exists(result_document):
        os.remove(result_document)

    with open(result_document, 'w+') as f:
        f.write('train_time\t' + str(train_time) + '\n')
        temp_list = history.history['loss']
        length = len(temp_list)
        f.write('loss\t\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value]) + ',')
            else:
                f.write(str(temp_list[i_value]) + '\n')

        temp_list = history.history['mean_squared_error']
        length = len(temp_list)
        f.write('mean_squared_error\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value]) + ',')
            else:
                f.write(str(temp_list[i_value]) + '\n')

        temp_list = history.history['val_loss']
        length = len(temp_list)
        f.write('val_loss\t\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value]) + ',')
            else:
                f.write(str(temp_list[i_value]) + '\n')

        temp_list = history.history['val_mean_squared_error']
        length = len(temp_list)
        f.write('val_mean_squared_error\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value]) + ',')
            else:
                f.write(str(temp_list[i_value]) + '\n')

    start_predict_time = time.time()
    mse_0, _ = rbm.predict(0)
    classified_0 = model.predict(mse_0)
    classified_0 = np.array([1 if classified_0[i_value] > 0 else 0 for i_value in np.arange(len(classified_0))])

    mse_1, _ = rbm.predict(1)
    classified_1 = model.predict(mse_1)
    classified_1 = np.array([1 if classified_1[i_value] > 0 else 0 for i_value in np.arange(len(classified_1))])

    mse_2, _ = rbm.predict(2)
    classified_2 = model.predict(mse_2)
    classified_2 = np.array([1 if classified_2[i_value] > 0 else 0 for i_value in np.arange(len(classified_2))])

    mse_3, _ = rbm.predict(3)
    classified_3 = model.predict(mse_3)
    classified_3 = np.array([1 if classified_3[i_value] > 0 else 0 for i_value in np.arange(len(classified_3))])

    mse_4, _ = rbm.predict(3)
    classified_4 = model.predict(mse_4)
    classified_4 = np.array([1 if classified_4[i_value] > 0 else 0 for i_value in np.arange(len(classified_4))])
    end_predict_time = time.time()
    predict_time = end_predict_time - start_predict_time
    classification = np.hstack((classified_0, classified_1, classified_2, classified_3, classified_4))

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    length = len(classification)
    label_positive = np.full(len(mse_0), 1)
    label_negative = np.zeros(len(mse_0) * 4)
    label = np.hstack((label_positive, label_negative))

    for i_value in np.arange(length):
        if label[i_value] == 1 and classification[i_value] == 1:
            TP = TP + 1
        elif label[i_value] == 1 and classification[i_value] == 0:
            TN = TN + 1
        elif label[i_value] == 0 and classification[i_value] == 1:
            FP = FP + 1
        elif label[i_value] == 0 and classification[i_value] == 0:
            FN = FN + 1

    result_document = result_directory + 'DBN_predict_result.txt'
    if os.path.exists(result_document):
        os.remove(result_document)

    with open(result_document, 'w+') as f:
        f.write('TP\t' + str(TP) + '\t' + 'TN\t' + str(TN) + '\t' + 'FP\t' + str(FP) + '\t' + 'FN\t' + '\t' +
                str(FN) + '\n')
        f.write('predict_time\t' + str(predict_time) + '\n')
        length = len(mse_0)
        f.write('mse_0\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(mse_0[i_value]) + ',')
            else:
                f.write(str(mse_0[i_value]) + '\n')

        length = len(mse_1)
        f.write('mse_1\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(mse_0[i_value]) + ',')
            else:
                f.write(str(mse_0[i_value]) + '\n')

        length = len(mse_2)
        f.write('mse_2\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(mse_0[i_value]) + ',')
            else:
                f.write(str(mse_0[i_value]) + '\n')

        length = len(mse_3)
        f.write('mse_3\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(mse_0[i_value]) + ',')
            else:
                f.write(str(mse_0[i_value]) + '\n')

        length = len(mse_4)
        f.write('mse_4\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(mse_0[i_value]) + ',')
            else:
                f.write(str(mse_0[i_value]) + '\n')
