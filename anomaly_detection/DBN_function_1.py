import time
import torch.nn as nn
from IO_function import *
from sklearn.preprocessing import MinMaxScaler


class RBM(object):
    def __init__(self, train_name, n_visible=25, n_hidden=500, read_parameters=None, weight=None, h_bias=None,
                 v_bias=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.scale_model = None
        self.read_parameters = None
        self.train_name = train_name

        if read_parameters is None:
            persistent = np.random.uniform(0, 1, (1, n_hidden))
            self.persistent = torch.tensor(persistent, dtype=torch.float64)

            if weight is None:
                bounds = -4.0 * np.sqrt(6.0 / (self.n_visible + self.n_hidden))
                weight = np.random.uniform(-bounds, bounds, (self.n_visible, self.n_hidden))
                self.weight = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
            if h_bias is None:
                h_bias = np.random.uniform(0, 1, (self.n_hidden, ))
                self.h_bias = torch.tensor(h_bias, dtype=torch.float64, requires_grad=True)
            if v_bias is None:
                v_bias = np.random.uniform(0, 1, (self.n_visible,))
                self.v_bias = torch.tensor(v_bias, dtype=torch.float64, requires_grad=True)
        elif read_parameters == "parameters":
            self.weight, self.h_bias, self.v_bias, self.persistent = read_parameters_function()

        self.params = [self.weight, self.h_bias, self.v_bias]

    def gibbs_hvh(self, h_sample) -> object:
        v1_value = torch.sigmoid(torch.matmul(h_sample, torch.t(self.weight)) + self.v_bias)
        v1_simple_ReLU = nn.ReLU(inplace=True)
        v1_sample = v1_simple_ReLU(torch.sign(v1_value - torch.Tensor(v1_value.shape).uniform_()))

        h1_value = torch.sigmoid(torch.matmul(v1_sample, self.weight) + self.h_bias)
        h1_simple_ReLU = nn.ReLU(inplace=True)
        h1_sample = h1_simple_ReLU(torch.sign(h1_value - torch.Tensor(h1_value.shape).uniform_()))

        return v1_sample, h1_sample

    def energy_function(self, v_sample) -> object:
        h_prediction = torch.matmul(v_sample, self.weight) + self.h_bias
        v_bias_term = torch.matmul(v_sample, torch.unsqueeze(self.v_bias, dim=1))
        hidden_term = torch.sum(torch.log(1.0 + torch.exp(h_prediction)), dim=1)
        result = torch.sum(-hidden_term - v_bias_term, dim=(0, 1))

        return result

    def loss_function(self, input_value, actual_value) -> object:
        activation_h = torch.sigmoid(torch.matmul(input_value, self.weight) + self.h_bias)
        activation_v = torch.sigmoid(torch.matmul(activation_h, torch.t(self.weight)) + self.v_bias)

        v_scale = activation_v.detach().numpy()
        v = self.scale_model.inverse_transform(v_scale)
        v = torch.tensor(v, dtype=torch.float64)

        mse = torch.sqrt(torch.mean(torch.square(actual_value - v)))
        mape = 100 * torch.sum(torch.abs(actual_value - v) / actual_value) / actual_value.shape[0]

        return mse, mape, activation_h, activation_v, v

    def predict_function(self, predict_data, actual_data) -> object:
        activation_h = torch.sigmoid(torch.matmul(predict_data, self.weight) + self.h_bias)
        activation_v = torch.sigmoid(torch.matmul(activation_h, torch.t(self.weight)) + self.v_bias)

        v_scale = activation_v.detach().numpy()
        v = self.scale_model.inverse_transform(v_scale)
        v = torch.tensor(v, dtype=torch.float64)

        mse = torch.sqrt(torch.mean(torch.square(v - actual_data)))
        mape = 100 * torch.sum(torch.abs(v - actual_data)) / actual_data.shape[0]

        return mse, mape, activation_h, activation_v, v

    def train_step(self, input_value, k):
        learning_rate = 0.1
        h1_value = torch.sigmoid(torch.mm(input_value, self.weight, out=None) + self.h_bias)
        h1_sample_ReLU = nn.ReLU(inplace=True)
        h1_sample = h1_sample_ReLU(torch.sign(h1_value - torch.Tensor(h1_value.shape).uniform_()))

        if self.persistent is None:
            chain_start = h1_sample
        else:
            chain_start = self.persistent

        nv_sample = 0
        nh_sample = 0
        for _ in np.arange(k):
            nv_sample, nh_sample = self.gibbs_hvh(chain_start)
        chain_end = nv_sample.detach()

        cost = self.energy_function(input_value) - self.energy_function(chain_end)
        g_weight, g_h_bias, g_v_bias = torch.autograd.grad(outputs=cost, inputs=self.params, retain_graph=True)
        self.weight = self.weight - learning_rate * g_weight
        self.h_bias = self.h_bias - learning_rate * g_h_bias
        self.v_bias = self.v_bias - learning_rate * g_v_bias

        if self.persistent is not None:
            self.persistent = nh_sample
        else:
            self.persistent = None

    def fit(self, input_data, epochs, k):
        if input_data is None:
            input_values = torch.Tensor(2, self.n_visible).uniform_().double()
            validation = input_values[:2, ]
        else:
            self.scale_model = MinMaxScaler()
            input_values = self.scale_model.fit_transform(input_data)
            length = len(input_data)
            validation_length = int(0.2 * length)
            validation = torch.tensor(input_values[:validation_length, ], dtype=torch.float64)
            input_values = torch.tensor(input_values, dtype=torch.float64)
            input_data = torch.tensor(input_data, dtype=torch.float64)

        clear_file(self.train_name)

        input_length = input_values.shape[0]
        input_dimension = input_values.shape[1]
        validation_length = validation.shape[0]

        for epoch in range(epochs):
            mse_epochs = []
            mape_epochs = []

            input_iterator = data_slice_function(input_values)
            validation_iterator = data_slice_function(validation)
            actual_iterator = data_slice_function(input_data)
            validation_actual_iterator = data_slice_function(input_data[:validation_length, ])

            train_start_time = time.time()
            for _ in np.arange(input_length):
                input_value = torch.reshape(next(input_iterator), (1, input_dimension))
                actual_value = torch.reshape(next(actual_iterator), (1, input_dimension))
                self.train_step(input_value, k)
                mse_epoch, mape_epoch, _, _, _ = self.loss_function(input_value, actual_value)
                mse_epochs.append(mse_epoch.item())
                mape_epochs.append(mape_epoch.item())
            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            mse = np.mean(mse_epochs)
            mape = np.mean(mape_epochs)

            val_mse_epochs = []
            val_mape_epochs = []

            for _ in np.arange(validation_length):
                validation_value = torch.reshape(next(validation_iterator), (1, input_dimension))
                validation_actual_value = torch.reshape(next(validation_actual_iterator), (1, input_dimension))
                val_mse_epoch, val_mape_epoch, _, _, _ = self.loss_function(validation_value, validation_actual_value)
                val_mse_epochs.append(val_mse_epoch.item())
                val_mape_epochs.append(val_mape_epoch.item())

            val_mse = np.mean(val_mse_epochs)
            val_mape = np.mean(val_mape_epochs)
            record_list = [mse, mape, val_mse, val_mape, train_time]
            string_print = "epoch = {1}/{0} \t mse = {2} \t mape = {3} \t val_mse = {4} \t val_mape = {5} \t " \
                           "train_time = {6}"
            print(string_print.format(str(epochs), str(epoch + 1), str(mse), str(mape), str(val_mse), str(val_mape),
                                      str(train_time)))
            write_to_text(self.train_name, record_list, "a+")

            params = self.params + [self.persistent]
            write_parameter_to_text(params, self.train_name)

    def predict(self, predict_values, model, file_name):
        predict_scale = self.scale_model.fit_transform(predict_values)
        predict_scale = torch.tensor(predict_scale, dtype=torch.float64)
        predict_values = torch.tensor(predict_values, dtype=torch.float64)

        predict_iterator = data_slice_function(predict_scale)
        actual_predict_iterator = data_slice_function(predict_values)

        predict_length = predict_values.shape[0]
        predict_dimension = predict_values.shape[1]

        activate_hs = []

        if model == "point":
            clear_file(file_name)
            for _ in np.arange(predict_length):
                predict_value = torch.reshape(next(predict_iterator), (1, predict_dimension))
                actual_value = torch.reshape(next(actual_predict_iterator), (1, predict_dimension))
                predict_start_time = time.time()
                mse_epoch, mape_epoch, activate_h, _, _ = self.predict_function(predict_value, actual_value)
                predict_end_time = time.time()
                predict_epoch_time = predict_end_time - predict_start_time

                out = activate_h.detach().numpy().squeeze()
                activate_hs.append(out)
                record_list = [mse_epoch.item(), mape_epoch.item(), predict_epoch_time] + out.tolist()
                write_to_text(file_name, record_list, "a+")
        else:
            clear_file(file_name)
            for _ in np.arange(predict_length):
                predict_value = torch.reshape(next(predict_iterator), (1, predict_dimension))
                actual_value = torch.reshape(next(actual_predict_iterator), (1, predict_dimension))
                predict_start_time = time.time()
                mse_epoch, mape_epoch, activate_h, _, v = self.predict_function(predict_value, actual_value)
                predict_end_time = time.time()
                predict_epoch_time = predict_end_time - predict_start_time
                v = v.numpy().squeeze().tolist()
                activate_hs.append(activate_h.detach().numpy())
                record_list = [mse_epoch.item(), mape_epoch.item(), predict_epoch_time] + v
                write_to_text(file_name, record_list, "a+")

        activate_hs = np.array(activate_hs).squeeze()
        return activate_hs
