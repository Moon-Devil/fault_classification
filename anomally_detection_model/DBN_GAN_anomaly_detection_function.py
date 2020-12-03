import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time


learning_rate = 0.01
training_epochs = 20
inner_epochs = 3
display_step = 1
node_visible = 27
node_hidden = 27


class RBM_model(object):
    def __init__(self, input_value=None, n_visible=node_visible, n_hidden=node_hidden, weight=None, h_bias=None,
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
    def condition(i_value, inner_epochs, nv_value, nv_sample, nh_value, nh_sample) -> object:
        return i_value < inner_epochs

    def body(self, i_value, inner_epochs, nv_value, nv_sample, nh_value, nh_sample) -> object:
        i_value = i_value + 1
        nv_value, nv_sample, nh_value, nh_sample = self.gibbs_hvh(nh_sample)
        return i_value, inner_epochs, nv_value, nv_sample, nh_value, nh_sample

    def train_step(self, lr=learning_rate, in_epochs=1, persistent=None) -> object:
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
        g_param_single = tf.gradients(ys=[cost], xs=self.params)

        g_params = []
        for i in range(len(g_param_single)):
            g_params.append(tf.clip_by_value(g_param_single[i], clip_value_min=-1, clip_value_max=1))

        new_params = []
        for g_param, param in zip(g_params, self.params):
            new_params.append(tf.assign(param, param - g_param * lr))

        if persistent is not None:
            new_persistent = [tf.assign(persistent, nh_sample)]
        else:
            new_persistent = []

        return new_params + new_persistent

    def get_reconstruction_cost(self) -> object:
        activation_h = tf.keras.activations.sigmoid(tf.matmul(self.input_value, self.weight) + self.h_bias)
        activation_v = tf.nn.sigmoid(tf.matmul(activation_h, tf.transpose(self.weight)) + self.v_bias)
        activation_v_clip = tf.clip_by_value(activation_v, clip_value_min=1e-30, clip_value_max=1)
        reduce_activation_v_clip = tf.clip_by_value(1.0 - activation_v, clip_value_min=1e-30, clip_value_max=1)

        cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input_value * tf.log(activation_v_clip)) +
                                        (1.0 - self.input_value) * (tf.log(reduce_activation_v_clip)), axis=1)
        return cross_entropy, activation_h

    def reconstruction(self, inputs):
        hidden_value = tf.keras.activations.sigmoid(tf.matmul(inputs, self.weight) + self.h_bias)
        visible_value = tf.keras.activations.sigmoid(tf.matmul(hidden_value, tf.transpose(self.weight)) + self.v_bias)
        return visible_value

    def train(self):
        train_param = self.train_step(lr=learning_rate, inner_epochs=inner_epochs)
        cost = self.get_reconstruction_cost()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)


print('done...')
