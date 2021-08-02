from IO_function import *
import time


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dimension):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 2),
            torch.nn.ReLU(inplace=True)
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
        out = self.decoder(middle)

        return out


class Decoder(torch.nn.Module):
    def __init__(self, input_dimension):
        super(Decoder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, input_dimension),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x) -> object:
        out = self.model(x)
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, input_dimension):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x) -> object:
        out = self.model(x)
        return out


class GAN(torch.nn.Module):
    def __init__(self, input_dimension):
        super(GAN, self).__init__()
        self.auto_encoder = AutoEncoder(input_dimension)
        self.decoder = Decoder(input_dimension)
        self.discriminator = Discriminator(input_dimension)

    @staticmethod
    def loss(calculations, true) -> object:
        loss = 100 * torch.sum(torch.abs(calculations - true) / true) / true.shape[0]
        mse = torch.sqrt(torch.mean(torch.square(calculations - true)))

        return loss, mse

    @staticmethod
    def discriminator_loss(calculations, true) -> object:
        loss_function = torch.nn.BCELoss()
        loss = loss_function(calculations, true)
        mse = torch.sqrt(torch.mean(torch.square(calculations - true)))

        return loss, mse

    def train_auto_encoder(self, x, epochs):
        clear_file("GAN_auto_encoder_train_curve")
        x_tensor = torch.tensor(x, dtype=torch.float32)
        validation_x = x_tensor[:301, ]

        optimizer_auto_encoder = torch.optim.Adam(self.auto_encoder.parameters(), lr=0.01)

        for epoch in np.arange(epochs):
            x_tensor.to("cuda:0")

            start_train_time = time.time()
            optimizer_auto_encoder.zero_grad()
            output = self.auto_encoder(x_tensor)
            loss, mse = self.loss(output, x_tensor)
            loss.backward()
            optimizer_auto_encoder.step()
            end_train_time = time.time()
            train_time = end_train_time - start_train_time

            val_output = self.auto_encoder(validation_x)
            val_loss, val_mse = self.loss(val_output, validation_x)
            string_print = "AutoEncoder: epoch = {1}/{0} \t loss = {2} \t mse = {3} \t val_loss = {4} \t " \
                           "val_mse = {5} \t train_time = {6}"
            print(string_print.format(str(epochs), str(epoch + 1), str(loss.item()), str(mse.item()),
                                      str(val_loss.item()), str(val_mse.item()), str(train_time)))
            record_list = [loss.item(), mse.item(), val_loss.item(), val_mse.item(), train_time]
            write_to_text("GAN_auto_encoder_train_curve", record_list, "a+")

    def train_GAN(self, x, epochs):
        real = np.full((len(x), 1), 1)
        fake = np.zeros((len(x), 1))

        real = torch.as_tensor(torch.from_numpy(real), dtype=torch.float32)
        fake = torch.as_tensor(torch.from_numpy(fake), dtype=torch.float32)

        clear_file("GAN_train_curve")
        x_tensor = torch.tensor(x, dtype=torch.float32)
        validation_x = x_tensor[:301, ]

        optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=0.01)
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.01)

        x_tensor.to("cuda:0")
        encoder_value = self.auto_encoder.encoder(x_tensor)

        for epoch in np.arange(epochs):
            start_train_time = time.time()
            optimizer_decoder.zero_grad()
            decoder_values = self.decoder(encoder_value)
            decoder_loss, decoder_mse = self.loss(decoder_values, x_tensor)
            decoder_loss.backward(retain_graph=True)
            optimizer_decoder.step()

            optimizer_discriminator.zero_grad()
            real_values = self.discriminator(x_tensor)
            fake_values = self.discriminator(decoder_values.detach())

            real_loss, real_mse = self.discriminator_loss(real_values, real)
            fake_loss, fake_mse = self.discriminator_loss(fake_values, fake)

            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            optimizer_discriminator.step()
            end_train_time = time.time()

            discriminator_mse = (real_mse + fake_mse) / 2
            train_time = end_train_time - start_train_time

            validation_x.to("cuda:0")
            val_encoder_values = self.auto_encoder.encoder(validation_x)
            val_decoder_values = self.decoder(val_encoder_values)
            val_loss, val_mse = self.loss(val_decoder_values, validation_x)

            string_print = "GAN: epoch = {1}/{0} \t loss = {2} \t mse = {3} \t val_loss = {4} \t val_mse = {5} \t " \
                           "train_time = {6}"
            print(string_print.format(str(epochs), str(epoch + 1), str(discriminator_loss.item()),
                                      str(discriminator_mse.item()), str(val_loss.item()), str(val_mse.item()),
                                      str(train_time)))
            record_list = [discriminator_loss.item(), discriminator_mse.item(), val_loss.item(), val_mse.item(),
                           train_time]
            write_to_text("GAN_train_curve", record_list, "a+")

    def fit(self, x, epochs):
        self.train_auto_encoder(x, epochs)
        self.train_GAN(x, epochs)

    def predict(self, x, model, filename):
        clear_file(filename)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        if model == "point":
            x_tensor.to("cuda:0")
            encoder_value = self.auto_encoder.encoder(x_tensor)
            encoder_value = encoder_value.detach().numpy().tolist()
            length = len(encoder_value)
            for i_index in np.arange(length):
                write_to_text(filename, encoder_value[i_index], "a+")
        elif model == "line":
            x_tensor.to("cuda:0")
            start_reconstruct_time = time.time()
            encoder_value = self.auto_encoder.encoder(x_tensor)
            reconstruct_value = self.decoder(encoder_value)
            end_reconstruct_time = time.time()

            reconstruct_value = reconstruct_value.detach().numpy().tolist()

            reconstruct_time = end_reconstruct_time - start_reconstruct_time
            write_to_text(filename, [reconstruct_time], "a+")

            length = len(reconstruct_value)
            for i_index in np.arange(length):
                write_to_text(filename, reconstruct_value[i_index], "a+")
