import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

import pickle
import gzip


class LSTM(nn.Module):

    def __init__(self, input_size=28, output_size=10, bias=True):
        super(LSTM, self).__init__()

        self.cell = LSTMCell(input_size=input_size,
                             output_size=output_size,
                             bias=bias
                             )

    def forward(self, *x):

        image = x.reshape([-1, 28, 28])

        output = None

        for i in range(28):

            output = self.cell(image[:, i, :])

        return output


class LSTMCell(nn.Module):
    """
    Basic LSTM Cell that can be reused as any other block. The code was written with this blog as a guide for the
    internal functions and equations:

    https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, input_size, output_size, bias=True):
        super(LSTMCell, self).__init__()

        self.cell_state = torch.zeros(output_size).float()
        self.hidden_layer = torch.zeros(output_size).float()

        # self.cell_state.reshape([-1, 28, 28])

        self.forget_gate = nn.Sequential(nn.Linear(in_features=input_size + output_size,
                                                   out_features=output_size,
                                                   bias=bias),
                                         nn.Sigmoid()
                                         )

        self.input_gate = nn.Sequential(nn.Linear(in_features=input_size + output_size,
                                                  out_features=output_size,
                                                  bias=bias),
                                        nn.Sigmoid()
                                        )

        self.new_cell_state = nn.Sequential(nn.Linear(in_features=input_size + output_size,
                                                      out_features=output_size,
                                                      bias=bias),
                                            nn.Tanh()
                                            )

        self.output_gate = nn.Sequential(nn.Linear(in_features=input_size+output_size,
                                                   out_features=output_size,
                                                   bias=bias),
                                         nn.Sigmoid()
                                         )

        self.tanh_act = nn.Tanh()

    def forward(self, *x):

        gate_input = torch.cat([self.hidden, x], dim=0)  # TODO confirm if the dimension matches.

        intrim_cell_state = self.cell_state * self.forget_gate(gate_input)

        self.cell_state = intrim_cell_state + (self.input_gate(gate_input) * self.new_cell_state(gate_input))

        self.hidden_layer = self.output_gate(gate_input) * self.tanh_act(self.cell_state)

        return self.hidden_layer


class MNISTDataset(Dataset):

    def __init__(self, mode="TRAIN", transform=T.ToTensor()):

        filename = 'mnist.pkl.gz'
        f = gzip.open(filename, 'rb')
        train, val, test = pickle.load(f, encoding='latin1')
        f.close()

        self.tf = transform

        # self.train_data = train[0]
        # self.train_label = train[1]
        #
        # self.val_data = val[0]
        # self.val_label = val[1]
        #
        # self.test_data = test[0]
        # self.test_label = test[1]

        if mode == "TEST":
            self.data = test[0]
            self.label = test[1]
        elif mode == "VAL":
            self.data = val[0]
            self.label = val[1]
        else:
            self.data = train[0]
            self.label = train[1]

        # train, val, test = None, None, None

        self.length = self.data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        digit = self.data[item, :]
        label = self.label[item]

        return self.tf(digit), self.tf(label)
