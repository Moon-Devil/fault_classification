import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import os


father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)


class RBM_model(object):
    def __init__(self, input_value=None, n_visible=27, n_hidden=27, weight=None, h_bias=None,
                 v_bias=None):
        self.n_visible = tf.constant(n_visible)
        self.n_hidden = tf.constant(n_hidden)
        self.prob = tf.constant(1)
        self.input_value = tf.cast(tf.constant(input_value), dtype=tf.float32)

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
        h1_value = tf.keras.activations.sigmoid(tf.matmul(self.input_value, self.weight) + self.h_bias)
        h1_sample = tf.keras.activations.relu(tf.sign(h1_value - tf.random_uniform(tf.shape(h1_value))))
        if persistent is None:
            chain_start = h1_sample
        else:
            chain_start = persistent

        i_value, in_epochs, nv_value, nv_sample, nh_value, nh_sample = \
            tf.while_loop(self.condition, self.body,
                          loop_vars=[tf.constant(0), tf.constant(in_epochs),
                                     tf.zeros(tf.shape(self.input_value)), tf.zeros(tf.shape(self.input_value)),
                                     tf.zeros(tf.shape(chain_start)), chain_start])

        update_weight = self.weight + lr * (tf.matmul(tf.transpose(self.input_value), h1_value) -
                                            tf.matmul(tf.transpose(nv_sample), nh_value)) / tf.to_float(
            tf.shape(self.input_value)[0])
        update_v_bias = self.v_bias + lr * (tf.reduce_mean(self.input_value - nv_sample, axis=0))
        update_h_bias = self.h_bias + lr * (tf.reduce_mean(h1_value - nh_value, axis=0))

        self.weight = update_weight
        self.v_bias = update_v_bias
        self.h_bias = update_h_bias

        chain_end = tf.stop_gradient(nv_sample)
        cost = tf.reduce_mean(self.energy_function(self.input_value)) - tf.reduce_mean(self.energy_function(chain_end))
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
        activation_h = tf.keras.activations.sigmoid(tf.matmul(self.input_value, self.weight) + self.h_bias)
        activation_v = tf.keras.activations.sigmoid(tf.matmul(activation_h, tf.transpose(self.weight)) + self.v_bias)
        mean_squared_error = tf.sqrt(tf.reduce_mean(tf.square(self.input_value - activation_v)))

        return mean_squared_error, activation_h, activation_v

    def predict(self, inputs) -> object:
        inputs = tf.cast(tf.constant(inputs), dtype=tf.float32)
        activation_h = tf.keras.activations.sigmoid(tf.matmul(inputs, self.weight) + self.h_bias)
        activation_v = tf.keras.activations.sigmoid(tf.matmul(activation_h, tf.transpose(self.weight)) + self.v_bias)
        mean_squared_error_tensor = tf.keras.metrics.mean_squared_error(inputs, activation_v)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            mean_squared_error = sess.run(mean_squared_error_tensor)

        return mean_squared_error

    def train(self, learning_rate, training_epochs, inner_epochs) -> object:
        loss = []
        cost_time = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(training_epochs):
                epoch_start_time = time.time()
                self.train_step(lr=learning_rate, in_epochs=inner_epochs)
                loss_tensor, _, _ = self.loss_function()
                epoch_loss = sess.run(loss_tensor)
                loss.append(epoch_loss)
                epoch_end_time = time.time()
                epoch_cost_time = epoch_end_time - epoch_start_time
                cost_time.append(epoch_cost_time)
                log = "epoch={0}, loss={1}, time={2}"
                print(log.format(epoch, epoch_loss, epoch_cost_time))

            _, hidden_outputs, visible_outputs = self.loss_function()
            hidden_outputs = sess.run(hidden_outputs)
            visible_outputs = sess.run(visible_outputs)

        return hidden_outputs, visible_outputs


def RBM_train(x_data, x_test, node_visible, node_hidden, learning_rate, training_epochs, inner_epochs, flag) -> object:
    scale_model = MinMaxScaler()
    x_train = scale_model.fit_transform(x_data)
    rbm = RBM_model(x_train, node_visible, node_hidden)
    hidden_result, visible_predict_scale = rbm.train(learning_rate, training_epochs, inner_epochs)
    visible_predict = scale_model.inverse_transform(visible_predict_scale)

    mean_squared_error_tensor = tf.keras.metrics.mean_squared_error(x_data, visible_predict)
    mean_squared_error_param_tensor = tf.sqrt(tf.reduce_mean(tf.square(x_data - visible_predict), axis=0))
    with tf.Session() as sess:
        mean_squared_error = sess.run(mean_squared_error_tensor)
        mean_squared_error_param = sess.run(mean_squared_error_param_tensor)

    mean_squared_error_predict = rbm.predict(x_test)
    return mean_squared_error_predict
