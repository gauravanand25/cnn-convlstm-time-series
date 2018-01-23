import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as torch_utils
from torch.autograd import Variable

from utils import plot_losses

class ConvAutoEncoder(nn.Module):
    def __init__(self, hidden_channels, kernel_sizes, strides, padding):
        super(ConvAutoEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding
        self.encode = nn.Sequential(
            # conv channel = 1, H = 1, W = 512
            nn.Conv1d(out_channels=self.hidden_channels[0], in_channels=1, kernel_size=self.kernel_sizes[0],
                      stride=self.strides[0], padding=self.padding[0]),

            nn.Conv1d(out_channels=hidden_channels[1], in_channels=hidden_channels[0], kernel_size=self.kernel_sizes[1],
                      stride=self.strides[1], padding=self.padding[1]),

            nn.Conv1d(out_channels=hidden_channels[2], in_channels=hidden_channels[1], kernel_size=self.kernel_sizes[2],
                      stride=self.strides[2], padding=self.padding[2]),
        )

        self.decode = nn.Sequential(
            # channel = 16, H = N/A, W = 16
            nn.ConvTranspose1d(out_channels=self.hidden_channels[1], in_channels=hidden_channels[2],
                               kernel_size=self.kernel_sizes[2], stride=self.strides[2], padding=self.padding[2]),

            nn.ConvTranspose1d(out_channels=self.hidden_channels[0], in_channels=self.hidden_channels[1],
                               kernel_size=self.kernel_sizes[1], stride=self.strides[1],
                               padding=self.padding[1]),

            nn.ConvTranspose1d(out_channels=1, in_channels=self.hidden_channels[0], kernel_size=self.kernel_sizes[0],
                               stride=self.strides[0], padding=self.padding[0])
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def embeddings(self, data):
        N, L = data.shape
        batch = torch.from_numpy(data).contiguous()
        batch = Variable(batch.float(), requires_grad=False)
        X = self.encode(batch.view(N, 1, L))
        X = X.data.numpy().reshape(N, -1)
        return X


def fit(args, data, val_data, model):
    # dtype = torch.FloatTensor or #change to torch.cuda.FloatTensor to make it run on GPU

    dataloader = torch_utils.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch_utils.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    plot_loss = []
    plot_val_loss = []

    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            N, L = batch.shape
            batch = Variable(batch.float(), requires_grad=False)
            output = model(batch.view(N, 1, L))
            loss = criterion(output, target=batch)
            if i % 100 == 0:
                val_batch = next(iter(val_dataloader))
                val_batch = Variable(val_batch.float(), requires_grad=False)
                val_output = model(val_batch.view(N, 1, L))
                val_loss = criterion(val_output, target=val_batch)
                plot_loss.append(loss.data.numpy()[0])
                plot_val_loss.append(val_loss.data.numpy()[0])
                print 'epoch', epoch, 'num', i, 'loss', loss.data.numpy()[0], 'val loss', val_loss.data.numpy()[0]

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

    plot_losses(plot_loss, plot_val_loss)