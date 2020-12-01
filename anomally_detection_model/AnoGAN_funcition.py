import tensorflow as tf
import numpy as np


class RBM(object):
    def __init__(self, input_value=None, n_visible=500, n_hidden=4, w=None, h_bias=None, v_bias=None):
        self.n_visible = tf.constant(n_visible)
        self.n_hidden = tf.constant(n_hidden)
        self.prob = tf.constant(1)
        self.input_value = tf.cast(tf.constant(input_value), dtype=tf.float32)

        if w is None:
            bounds = -4.0 * np.sqrt(6.0 / (n_visible + n_hidden))
            w = tf.random_uniform([n_visible, n_hidden], minval=-bounds, maxval=bounds)
        if h_bias is None:
            h_bias = tf.zeros((self.n_hidden,))
        if v_bias is None:
            v_bias = tf.zeros((self.n_visible,))
        self.w = tf.convert_to_tensor(w)
        self.h_bias = tf.convert_to_tensor(h_bias)
        self.v_bias = tf.convert_to_tensor(v_bias)
        self.params = [self.w, self.h_bias, self.v_bias]

    def visible_to_hidden(self, visible) -> object:
        visible = tf.cast(tf.convert_to_tensor(visible), dtype=tf.float32)
        return tf.keras.activations.sigmoid(tf.matmul(visible, self.w) + self.h_bias)

    def hidden_to_visible(self, hidden) -> object:
        hidden = tf.cast(tf.convert_to_tensor(hidden), dtype=tf.float32)
        return tf.keras.activations.sigmoid(tf.matmul(hidden, tf.transpose(self.w)) + self.v_bias)

    @staticmethod
    def sample_prob(prob) -> object:
        return tf.keras.activations.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))

    def sample_h_given_v(self, v0_sample) -> object:
        h1_mean = self.hidden_to_visible(v0_sample)
        h1_sample = self.sample_prob(h1_mean)
        return h1_mean, h1_sample

    def sample_v_given_h(self, h0_sample) -> object:
        v1_mean = self.visible_to_hidden(h0_sample)
        v1_sample = self.sample_prob(v1_mean)
        return v1_mean, v1_sample

    def gibbs_vhv(self, v0_sample) -> object:
        h1_mean, h1_sample = self.sample_v_given_h(v0_sample)
        v1_mean, v1_sample = self.sample_h_given_v(h1_sample)
        return h1_mean, h1_sample, v1_mean, v1_sample

    def gibbs_hvh(self, h0_sample) -> object:
        v1_mean, v1_sample = self.sample_h_given_v(h0_sample)
        h1_mean, h1_sample = self.sample_v_given_h(v1_sample)
        return v1_mean, v1_sample, h1_mean, h1_sample

    def free_energy(self, v_sample) -> object:
        wx_b = tf.matmul(v_sample, self.w) + self.h_bias
        v_bias_term = tf.matmul(v_sample, tf.expand_dims(self.v_bias, axis=1))
        hidden_term = tf.reduce_sum(tf.log(1.0 + tf.exp(wx_b)), axis=1)
        return -hidden_term - v_bias_term

    @staticmethod
    def cond(i_value, k, nv_mean, nv_sample, nh_mean, nh_sample) -> object:
        return i_value < k

    def body(self, i_value, k, nv_mean, nv_sample, nh_mean, nh_sample) -> object:
        i_value = i_value + 1
        nv_mean, nv_sample, nh_mean, nh_sample = self.gibbs_hvh(nh_sample)
        return i_value, k, nv_mean, nv_sample, nh_mean, nh_sample

    def get_train_ops(self, learning_rate=0.1, k=1, persistent=None) -> object:
        ph_mean, ph_sample = self.sample_v_given_h(self.input_value)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        i, k, nv_mean, nv_sample, nh_mean, nh_sample = \
            tf.while_loop(self.cond, self.body,
                          loop_vars=[tf.constant(0), tf.constant(k), tf.zeros(tf.shape(self.input_value)),
                                     tf.zeros(tf.shape(self.input_value)), tf.zeros(tf.shape(chain_start)),
                                     chain_start])

        update_w = self.w + learning_rate * (tf.matmul(tf.transpose(self.input_value), ph_mean) -
                                             tf.matmul(tf.transpose(nv_sample), nh_mean)) / tf.to_float(
            tf.shape(self.input_value)[0])

        update_v_bias = self.v_bias + learning_rate * (tf.reduce_mean(self.input_value - nv_sample, axis=0))
        update_h_bias = self.h_bias + learning_rate * (tf.reduce_mean(ph_mean - nh_mean, axis=0))

        self.w = update_w
        self.v_bias = update_v_bias
        self.h_bias = update_h_bias

        chain_end = tf.stop_gradient(nv_sample)
        cost = tf.reduce_mean(self.free_energy(self.input_value)) - tf.reduce_mean(self.free_energy(chain_end))
        g_params = tf.gradients(ys=[cost], xs=self.params)

        new_params = []
        for g_params, param in zip(g_params, self.params):
            param = param - g_params * learning_rate
            new_params.append(param)

        if persistent is not None:
            persistent = nh_sample
            new_persistent = [persistent]
        else:
            new_persistent = []

        return new_params + new_persistent

    def get_reconstruction_cost(self) -> object:
        activation_h, _ = self.sample_v_given_h(self.input_value)
        activation_v, _ = self.sample_h_given_v(activation_h)

        activation_v_clip = tf.clip_by_value(activation_v, clip_value_min=1e-30, clip_value_max=1.0)
        reduce_activation_v_clip = tf.clip_by_value(1.0 - activation_v, clip_value_min=1e-30, clip_value_max=1.0)

        cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input_value * tf.log(activation_v_clip)) +
                                        (1.0 - self.input_value) * (tf.log(reduce_activation_v_clip)), axis=1)
        return cross_entropy, activation_h

    def reconstruct(self, v):
        h = self.visible_to_hidden(v)
        return self.hidden_to_visible(h)