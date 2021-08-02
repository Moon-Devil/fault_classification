import time

import torch.autograd

from IO_function import *
from sklearn.preprocessing import MinMaxScaler


class RBM(object):
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weight = None
        self.h_bias = None
        self.v_bias = None
        self.persistent = None
        self.scale_model = MinMaxScaler()

    def get_parameters(self, read_parameters) -> object:
        weight = 0
        h_bias = 0
        v_bias = 0
        persistent = 0

        if read_parameters is None:
            persistent = None
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visible + self.n_hidden))
            weight = np.random.uniform(-bounds, bounds, (self.n_visible, self.n_hidden))
            h_bias = np.random.rand(self.n_hidden)
            v_bias = np.random.rand(self.n_visible)
        elif read_parameters == "parameters":
            weight, h_bias, v_bias, persistent = read_parameters_function("RBM")

        return weight, h_bias, v_bias, persistent

    def gibbs_hvh(self, h_sample) -> object:
        v1_value = torch.sigmoid(torch.matmul(h_sample, torch.t(self.weight)) + self.v_bias)
        v1_simple_ReLU = torch.nn.ReLU(inplace=True)
        v1_sample = v1_simple_ReLU(torch.sign(v1_value - torch.rand(v1_value.shape)))

        h1_value = torch.sigmoid(torch.matmul(v1_sample, self.weight) + self.h_bias)
        h1_simple_ReLU = torch.nn.ReLU(inplace=True)
        h1_sample = h1_simple_ReLU(torch.sign(h1_value - torch.rand(h1_value.shape)))

        return v1_sample, h1_sample

    def energy_function(self, v_sample) -> object:
        h_prediction = torch.matmul(v_sample, self.weight) + self.h_bias
        v_bias_term = torch.matmul(v_sample, torch.unsqueeze(self.v_bias, dim=1))
        v_bias_term = v_bias_term.squeeze(dim=1)
        hidden_term = torch.mean(torch.log(1.0 + torch.exp(h_prediction)), dim=1)
        result = -hidden_term - v_bias_term

        return result

    def train_step(self, input_value, k):
        learning_rate = 0.01
        input_value.to("cuda:0")
        h1_value = torch.sigmoid(torch.mm(input_value, self.weight, out=None) + self.h_bias)
        h1_sample_ReLU = torch.nn.ReLU(inplace=True)
        h1_sample = h1_sample_ReLU(torch.sign(h1_value - torch.rand(h1_value.shape)))

        if self.persistent is None:
            chain_start = h1_sample
        else:
            chain_start = self.persistent

        hk_sample = 0
        vk_sample = 0
        for _ in np.arange(k):
            vk_sample, hk_sample = self.gibbs_hvh(chain_start)
            chain_start = hk_sample

        chain_end = vk_sample.detach()

        cost = self.energy_function(input_value) - self.energy_function(chain_end)
        cost.data.zero_()
        cost.backward(torch.ones_like(cost))
        g_weight = self.weight.grad
        g_h_bias = self.h_bias.grad
        g_v_bias = self.v_bias.grad

        weight = self.weight - learning_rate * g_weight
        h_bias = self.h_bias - learning_rate * g_h_bias
        v_bias = self.v_bias - learning_rate * g_v_bias

        self.weight = weight.clone().detach().requires_grad_(True)
        self.h_bias = h_bias.clone().detach().requires_grad_(True)
        self.v_bias = v_bias.clone().detach().requires_grad_(True)
        if self.persistent is not None:
            self.persistent = hk_sample
        else:
            self.persistent = None

    def loss_function(self, input_values) -> object:
        activation_h = torch.sigmoid(torch.matmul(input_values, self.weight) + self.h_bias)
        activation_v = torch.sigmoid(torch.matmul(activation_h, torch.t(self.weight)) + self.v_bias)

        v_scale = activation_v.detach().numpy()
        v = self.scale_model.inverse_transform(v_scale)

        input_values = input_values.detach().numpy()
        input_values = self.scale_model.inverse_transform(input_values)
        loss = np.sqrt(np.mean(np.square(input_values - v)))

        return loss, activation_h, activation_v, v

    def fit(self, filename, input_data, epochs, k):
        clear_file(filename)
        input_values = self.scale_model.fit_transform(input_data)
        validation = torch.tensor(input_values[: 301, ], dtype=torch.float64)
        input_values = torch.tensor(input_values, dtype=torch.float64)

        weight, h_bias, v_bias, persistent = self.get_parameters(None)
        self.weight = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
        self.h_bias = torch.tensor(h_bias, dtype=torch.float64, requires_grad=True)
        self.v_bias = torch.tensor(v_bias, dtype=torch.float64, requires_grad=True)
        if persistent is not None:
            self.persistent = torch.tensor(persistent, dtype=torch.float64, requires_grad=False)
        else:
            self.persistent = None

        self.weight.to("cuda:0")
        self.h_bias.to("cuda:0")
        self.v_bias.to("cuda:0")
        input_values.to("cuda:0")

        for epoch in range(epochs):
            start_train_time = time.time()
            self.train_step(input_values, k)
            loss, _, _, _ = self.loss_function(input_values)
            end_train_time = time.time()
            train_time = end_train_time - start_train_time

            val_loss, _, _, _ = self.loss_function(validation)

            record_list = [loss, val_loss, train_time]
            string_print = "epoch = {1}/{0} \t loss = {2} \t val_loss = {3} \t train_time = {4}"
            print(string_print.format(str(epochs), str(epoch + 1), str(loss), str(val_loss), str(train_time)))
            write_to_text(filename, record_list, "a+")

    def predict(self, predict_values, model, file_name) -> object:
        predict_scale = self.scale_model.fit_transform(predict_values)
        predict_scale = torch.tensor(predict_scale, dtype=torch.float64)
        predict_values = torch.tensor(predict_values, dtype=torch.float64)

        predict_scale.to("cuda:0")
        predict_values.to("cuda:0")
        self.weight.to("cuda:0")
        self.h_bias.to("cuda:0")
        self.v_bias.to("cuda:0")

        start_predict_time = time.time()
        activation_h = torch.sigmoid(torch.matmul(predict_scale, self.weight) + self.h_bias)
        activation_v = torch.sigmoid(torch.matmul(activation_h, torch.t(self.weight)) + self.v_bias)
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        if model == "point":
            clear_file(file_name)
            h = activation_h.detach().numpy().tolist()
            length = len(h)
            for i_index in np.arange(length):
                write_to_text(file_name, h[i_index], "a+")
        elif model == "line":
            clear_file(file_name)
            write_to_text(file_name, [predict_time], "a+")

            v_scale = activation_v.detach().numpy()
            v = self.scale_model.inverse_transform(v_scale).tolist()
            length = len(v)
            for i_index in np.arange(length):
                write_to_text(file_name, v[i_index], "a+")
        else:
            h = activation_h.detach().numpy()
            return h

    def predict_mse(self, predict_values, file_name):
        predict_scale = self.scale_model.fit_transform(predict_values)
        predict_scale = torch.tensor(predict_scale, dtype=torch.float64)
        predict_values = torch.tensor(predict_values, dtype=torch.float64)

        predict_scale.to("cuda:0")
        predict_values.to("cuda:0")
        self.weight.to("cuda:0")
        self.h_bias.to("cuda:0")
        self.v_bias.to("cuda:0")

        start_predict_time = time.time()
        activation_h = torch.sigmoid(torch.matmul(predict_scale, self.weight) + self.h_bias)
        activation_v = torch.sigmoid(torch.matmul(activation_h, torch.t(self.weight)) + self.v_bias)
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        write_to_text(file_name, [predict_time], "a+")

        clear_file(file_name)
        v_scale = activation_v.detach().numpy()
        v = self.scale_model.inverse_transform(v_scale).tolist()
        predict_values = predict_values.numpy()

        columns = np.shape(predict_values)[0]
        rows = np.shape(predict_values)[1]
        for i_index in np.arange(columns):
            loss = 0
            index = 0
            length_epoch = np.shape(np.where(v[i_index]))[0]
            if length_epoch == 0:
                loss = 100 * np.sum(np.abs(v[i_index] - predict_values[i_index]) / predict_values[i_index])
            else:
                for row in np.arange(rows):
                    if predict_values[i_index][row] != 0:
                        loss_epoch = np.abs(v[i_index][row] - predict_values[i_index][row]) / \
                                     predict_values[i_index][row]
                        loss += loss_epoch
                        index += 1
                loss = loss / index

            mse = np.sqrt(np.mean(np.square(v[i_index] - predict_values[i_index])))

            write_to_text(file_name, [loss, mse], "a+")
            print("write: " + str(i_index) + "/" + str(columns + 1))
