import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import plot_losses


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) / 2
        self.conv = nn.Conv1d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels, kernel_size=self.kernel_size, stride=1,
                              padding=self.padding)

    def forward(self, input, h, c):
        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, A.size()[1] / self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

    @staticmethod
    def init_hidden(batch_size, hidden_channels, width):
        return (Variable(torch.zeros(batch_size, hidden_channels, width)),
                Variable(torch.zeros(batch_size, hidden_channels, width)))

    @staticmethod
    def init_input(batch_size, input_channels, width):
        return Variable(torch.zeros(batch_size, input_channels, width))


class EncoderConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, filter_sizes):
        super(EncoderConvLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.filter_sizes = filter_sizes
        self.lstm1 = ConvLSTMCell(input_channels=self.input_channels, hidden_channels=hidden_channels[0],
                                  kernel_size=self.filter_sizes[0])
        self.lstm2 = ConvLSTMCell(input_channels=hidden_channels[0], hidden_channels=hidden_channels[1],
                                  kernel_size=self.filter_sizes[1])
        self.lstm3 = ConvLSTMCell(input_channels=hidden_channels[1], hidden_channels=hidden_channels[2],
                                  kernel_size=self.filter_sizes[2])

    def forward(self, input, time_step, h1, c1, h2, c2, h3, c3):
        if time_step == 0:
            bsize, _, width = input.size()
            (h1, c1) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[0], width)
        h1, c1 = self.lstm1(input, h1, c1)

        if time_step == 0:
            bsize, _, width = h1.size()
            (h2, c2) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[1], width)
        h2, c2 = self.lstm2(h1, h2, c2)

        if time_step == 0:
            bsize, _, width = h2.size()
            (h3, c3) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[2], width)
        h3, c3 = self.lstm3(h2, h3, c3)
        return h1, c1, h2, c2, h3, c3


class DecoderConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, filter_sizes):
        super(DecoderConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.filter_sizes = filter_sizes
        self.lstm1 = ConvLSTMCell(input_channels=self.input_channels, hidden_channels=self.hidden_channels[0],
                                  kernel_size=self.filter_sizes[0])
        self.lstm2 = ConvLSTMCell(input_channels=self.hidden_channels[0], hidden_channels=self.hidden_channels[1],
                                  kernel_size=self.filter_sizes[1])
        self.lstm3 = ConvLSTMCell(input_channels=self.hidden_channels[1], hidden_channels=self.hidden_channels[2],
                                  kernel_size=self.filter_sizes[2])

    def forward(self, input, time_step, h1, c1, h2, c2, h3, c3):
        if time_step == 0:
            bsize, _, width = input.size()
            (h1, c1) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[0], width)
        else:
            bsize, _, width = h1.size()
            input = ConvLSTMCell.init_input(bsize, self.input_channels, width)

        h1, c1 = self.lstm1(input, h1, c1)

        if time_step == 0:
            bsize, _, width = h1.size()
            (h2, c2) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[1], width)
        h2, c2 = self.lstm2(h1, h2, c2)

        if time_step == 0:
            bsize, _, width = h2.size()
            (h3, c3) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[2], width)
        h3, c3 = self.lstm3(h2, h3, c3)

        return h1, c1, h2, c2, h3, c3


def fit(args, data_loader, val_data_loader):
    criterion = nn.MSELoss()

    encoder = EncoderConvLSTM(input_channels=1, hidden_channels=[36, 12, 4], filter_sizes=[21, 11, 7])
    decoder = DecoderConvLSTM(input_channels=4, hidden_channels=[12, 36, 1], filter_sizes=[7, 11, 21])

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    plot_loss = []
    plot_val_loss = []

    iterations = data_loader.get_length() / args.batch_size
    for epoch in range(args.epochs):
        for itr in range(iterations):
            batch, _ = data_loader.get_batch()
            val_batch, _ = val_data_loader.get_batch()  # for validation loss

            len = args.conv_len  # args.filter_size
            batch = data_loader.fix_batch_len(batch, len, zeros=True)
            val_batch = val_data_loader.fix_batch_len(val_batch, len, zeros=True)
            N, L = batch.shape

            inputs = []
            loss = 0
            h1_en, c1_en, h2_en, c2_en, h3_en, c3_en = None, None, None, None, None, None
            for i in range(0, L, len):
                batch_curr_time = Variable(torch.FloatTensor(batch[:, i:i + len]), requires_grad=False)
                h1_en, c1_en, h2_en, c2_en, h3_en, c3_en = encoder(batch_curr_time.view(N, 1, len), i / len, h1_en,
                                                                   c1_en, h2_en, c2_en, h3_en, c3_en)
                inputs.append(batch_curr_time)

            h1_dec, c1_dec, h2_dec, c2_dec, h3_dec, c3_dec = None, None, None, None, None, None
            # dec_input = h_en.clone()
            for i in range(0, L, len):
                h1_dec, c1_dec, h2_dec, c2_dec, h3_dec, c3_dec = decoder(h3_en if i == 0 else None, i / len, h1_dec,
                                                                         c1_dec, h2_dec, c2_dec, h3_dec, c3_dec)
                loss += criterion(h3_dec, target=inputs[L / len - 1 - i / len])

            loss /= L / 100.0  # normalize loss

            if itr % 100 == 0:
                N_val, L_val = val_batch.shape
                val_inputs = []
                val_loss = 0
                val_h1_en, val_c1_en, val_h2_en, val_c2_en, val_h3_en, val_c3_en = None, None, None, None, None, None
                for j in range(0, L_val, len):
                    val_batch_curr_time = Variable(torch.FloatTensor(val_batch[:, j:j + len]), requires_grad=False)
                    val_h1_en, val_c1_en, val_h2_en, val_c2_en, val_h3_en, val_c3_en = encoder(
                        val_batch_curr_time.view(N, 1, len), j / len,
                        val_h1_en, val_c1_en, val_h2_en,
                        val_c2_en, val_h3_en, val_c3_en)
                    val_inputs.append(val_batch_curr_time)

                val_h1_dec, val_c1_dec, val_h2_dec, val_c2_dec, val_h3_dec, val_c3_dec = None, None, None, None, None, None
                # dec_input = h_en.clone()
                for j in range(0, L_val, len):
                    val_h1_dec, val_c1_dec, val_h2_dec, val_c2_dec, val_h3_dec, val_c3_dec = decoder(
                        val_h3_en if j == 0 else None, j / len,
                        val_h1_dec, val_c1_dec, val_h2_dec,
                        val_c2_dec, val_h3_dec, val_c3_dec)
                    val_loss += criterion(val_h3_dec, target=val_inputs[L_val / len - 1 - j / len])

                val_loss /= L_val / 100.0
                plot_loss.append(loss.data.numpy()[0])
                plot_val_loss.append(val_loss.data.numpy()[0])
                print 'epoch', epoch, 'num', itr, 'loss', loss.data.numpy()[0], 'val loss', val_loss.data.numpy()[0]

            encoder_optimizer.zero_grad()  # zero the gradient buffers
            decoder_optimizer.zero_grad()

            loss.backward()

            encoder_optimizer.step()  # Does the update
            decoder_optimizer.step()

    plot_losses(plot_loss, plot_val_loss)
    return encoder, decoder


def embeddings(args, data_util, data, encoder):
    len = args.conv_len
    data = data_util.fix_batch_len(data, len, zeros=False)
    N, L = data.shape

    h1_en, c1_en, h2_en, c2_en, h3_en, c3_en = None, None, None, None, None, None
    for i in range(0, L, len):
        data_curr_time = Variable(torch.FloatTensor(data[:, i:i + len]), requires_grad=False)
        h1_en, c1_en, h2_en, c2_en, h3_en, c3_en = encoder(data_curr_time.view(N, 1, len), i / len, h1_en, c1_en, h2_en,
                                                           c2_en, h3_en, c3_en)

    return h3_en.data.numpy().reshape(N, -1)
