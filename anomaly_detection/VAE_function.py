from IO_function import *
import time
from sklearn.preprocessing import MinMaxScaler


class VAE(torch.nn.Module):
    def __init__(self, input_dimension):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True)
        )

        self.mean_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
        )

        self.variance_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 25)
        )

    @staticmethod
    def parameterization(mean, variance) -> object:
        standard = torch.exp(0.5 * variance)
        z = torch.randn(standard.size()) * standard + mean

        return z

    @staticmethod
    def KL_divergence_function(reconstruct, x, mean, variance):
        mse = torch.sqrt(torch.mean(torch.square(reconstruct - x)))
        KL_divergence = -0.5 * torch.sum(1 + variance - torch.exp(variance) - mean ** 2)
        loss = mse + KL_divergence

        return loss

    @staticmethod
    def loss_function(reconstruct_x, x, reconstruct_validation_x, validation_x) -> object:
        loss = 0
        index = 0
        length = np.shape(np.where(x))[1]
        if length == 0:
            loss = 100 * np.sum(np.abs(reconstruct_x - x) / x) / x.shape[0]
        else:
            columns = np.shape(x)[0]
            rows = np.shape(x)[1]
            for column in np.arange(columns):
                for row in np.arange(rows):
                    if x[column][row] != 0:
                        loss_epoch = np.abs(reconstruct_x[column][row] - x[column][row]) / x[column][row]
                        loss += loss_epoch
                        index += 1
            loss = loss / index

        mse = np.sqrt(np.mean(np.square(reconstruct_x - x)))

        val_loss = 0
        val_index = 0
        val_length = np.shape(np.where(validation_x))[1]
        if val_length == 0:
            val_loss = 100 * np.sum(np.abs(reconstruct_validation_x - validation_x) / validation_x) / \
                       validation_x.shape[0]
        else:
            columns = np.shape(validation_x)[0]
            rows = np.shape(validation_x)[1]
            for column in np.arange(columns):
                for row in np.arange(rows):
                    if validation_x[column][row] != 0:
                        val_loss_epoch = np.abs(reconstruct_x[column][row] - validation_x[column][row]) / \
                                         validation_x[column][row]
                        val_loss += val_loss_epoch
                        val_index += 1
            val_loss = val_loss / val_index

        val_mse = np.sqrt(np.mean(np.square(reconstruct_validation_x - validation_x)))

        return [loss, mse, val_loss, val_mse]

    def forward(self, x):
        encoder_values = self.encoder(x)
        mean = self.mean_layer(encoder_values)
        variance = self.variance_layer(encoder_values)
        sample_values = self.parameterization(mean, variance)
        output = self.decoder(sample_values)

        return output

    def fit(self, x, epochs):
        clear_file("VAE_train_curve")
        scale_model = MinMaxScaler()
        x_scale = scale_model.fit_transform(x)

        x_scale = torch.tensor(x_scale, dtype=torch.float32)
        validation_x = x_scale[: 301, ]

        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        for epoch in np.arange(epochs):
            x_scale.to("cuda:0")
            start_train_time = time.time()

            optimizer.zero_grad()
            encoder_values = self.encoder(x_scale)
            mean = self.mean_layer(encoder_values)
            variance = self.variance_layer(encoder_values)
            z = self.parameterization(mean, variance)
            reconstruct = self.decoder(z)
            DKL = self.KL_divergence_function(reconstruct, x_scale, mean, variance)
            DKL.backward()
            optimizer.step()

            end_train_time = time.time()
            train_time = end_train_time - start_train_time

            reconstruct_rescale = scale_model.inverse_transform(reconstruct.detach().numpy())
            val_reconstruct = self.forward(validation_x)

            val_reconstruct_rescale = scale_model.inverse_transform(val_reconstruct.detach().numpy())
            validation_x_rescale = scale_model.inverse_transform(validation_x.detach().numpy())
            loss = self.loss_function(reconstruct_rescale, x, val_reconstruct_rescale, validation_x_rescale)

            record_list = loss + [train_time]
            string_print = "epoch = {1}/{0} \t loss = {2} \t mse = {3} \t val_loss = {4} \t val_mse = {5} \t " \
                           "train_time = {6}"
            print(string_print.format(str(epochs), str(epoch + 1), str(loss[0]), str(loss[1]), str(loss[2]),
                                      str(loss[3]), str(train_time)))
            write_to_text("VAE_train_curve", record_list, "a+")

    def predict(self, predict_values, model, file_name) -> list:
        clear_file(file_name)
        scale_model = MinMaxScaler()
        predict_values_scale = scale_model.fit_transform(predict_values)
        predict_values_scale = torch.tensor(predict_values_scale, dtype=torch.float32)
        predict_values_scale.to("cuda:0")
        output = []

        if model == "point":
            encoder_values = self.encoder(predict_values_scale)
            mean = self.mean_layer(encoder_values)
            variance = self.variance_layer(encoder_values)
            mean = mean.detach().numpy()
            variance = variance.detach().numpy()

            length = len(mean)
            for i_index in np.arange(length):
                record_list = [mean[i_index][0], variance[i_index][0]]
                output.append(record_list)
                if file_name is not None:
                    write_to_text(file_name, record_list, "a+")

        elif model == "line":
            start_predict_time = time.time()
            output = self.forward(predict_values_scale)
            end_predict_time = time.time()
            predict_time = end_predict_time - start_predict_time

            output = output.detach().numpy()
            output = scale_model.inverse_transform(output)
            output = output.tolist()

            length = len(output)
            write_to_text(file_name, [predict_time], "a+")
            for i_index in np.arange(length):
                write_to_text(file_name, output[i_index], "a+")

        return output

    def predict_loss(self, predict_values, file_name):
        clear_file(file_name)
        scale_model = MinMaxScaler()
        predict_values_scale = scale_model.fit_transform(predict_values)
        predict_values_scale = torch.tensor(predict_values_scale, dtype=torch.float32)
        predict_values_scale.to("cuda:0")

        start_predict_time = time.time()
        output = self.forward(predict_values_scale)
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        output = output.detach().numpy()
        output = scale_model.inverse_transform(output)

        write_to_text(file_name, [predict_time], "a+")

        columns = np.shape(predict_values)[0]
        rows = np.shape(predict_values)[1]
        for i_index in np.arange(columns):
            loss = 0
            index = 0
            length_epoch = np.shape(np.where(output[i_index]))[0]
            if length_epoch == 0:
                loss = 100 * np.sum(np.abs(output[i_index] - predict_values[i_index]) / predict_values[i_index])
            else:
                for row in np.arange(rows):
                    if predict_values[i_index][row] != 0:
                        loss_epoch = np.abs(output[i_index][row] - predict_values[i_index][row]) / \
                                     predict_values[i_index][row]
                        loss += loss_epoch
                        index += 1
                loss = loss / index

            mse = np.sqrt(np.mean(np.square(output[i_index] - predict_values[i_index])))

            write_to_text(file_name, [loss, mse], "a+")
            print("write: " + str(i_index) + "/" + str(columns + 1))

    def reconstruction_function(self, input_values):
        scale_model = MinMaxScaler()
        predict_values_scale = scale_model.fit_transform(input_values)
        predict_values_scale = torch.tensor(predict_values_scale, dtype=torch.float32)
        predict_values_scale.to("cuda:0")

        output = self.forward(predict_values_scale)
        output = output.detach().numpy()
        output = scale_model.inverse_transform(output)

        return output
