from IO_function import *
import time


class Encoder(torch.nn.Module):
    def __init__(self, input_dimension):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 2),
            torch.nn.ReLU(inplace=True),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, input_dimension),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x) -> object:
        middle = self.encoder(x)
        output = self.decoder(middle)

        return output


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dimension):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dimension)

    def forward(self, x) -> object:
        output = self.encoder(x)

        return output

    @staticmethod
    def loss(calculations, true) -> object:
        loss = 100 * torch.sum(torch.abs(calculations - true) / true) / true.shape[0]
        mse = torch.sqrt(torch.mean(torch.square(calculations - true)))

        return loss, mse

    def fit(self, x, epochs):
        clear_file("AutoEncoder_train_curve")
        x_tensor = torch.tensor(x, dtype=torch.float32)
        validation_x = x_tensor[:301, ]

        optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.01)

        for epoch in np.arange(epochs):
            x_tensor.to("cuda:0")

            start_train_time = time.time()
            optimizer_encoder.zero_grad()
            output = self.encoder(x_tensor)
            loss, mse = self.loss(output, x_tensor)
            loss.backward()
            optimizer_encoder.step()
            end_train_time = time.time()

            train_time = end_train_time - start_train_time

            val_output = self.encoder(validation_x)
            val_loss, val_mse = self.loss(val_output, validation_x)

            string_print = "AutoEncoder: epoch = {1}/{0} \t loss = {2} \t mse = {3} \t val_loss = {4} \t " \
                           "val_mse = {5} \t train_time = {6}"
            print(string_print.format(str(epochs), str(epoch + 1), str(loss.item()), str(mse.item()),
                                      str(val_loss.item()), str(val_mse.item()), str(train_time)))
            record_list = [loss.item(), mse.item(), val_loss.item(), val_mse.item(), train_time]
            write_to_text("AutoEncoder_train_curve", record_list, "a+")

    def predict(self, x, model, filename):
        clear_file(filename)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        if model == "point":
            x_tensor.to("cuda:0")
            encoder_values = self.encoder.encoder(x_tensor)
            encoder_values = encoder_values.detach().numpy().tolist()
            length = len(encoder_values)
            for i_index in np.arange(length):
                write_to_text(filename, encoder_values[i_index], "a+")
        elif model == "line":
            x_tensor.to("cuda:0")
            start_reconstruct_time = time.time()
            reconstruct_value = self.encoder(x_tensor)
            end_reconstruct_time = time.time()

            reconstruct_value = reconstruct_value.detach().numpy().tolist()

            reconstruct_time = end_reconstruct_time - start_reconstruct_time
            write_to_text(filename, [reconstruct_time], "a+")

            length = len(reconstruct_value)
            for i_index in np.arange(length):
                write_to_text(filename, reconstruct_value[i_index], "a+")

