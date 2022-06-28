import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BasicBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_size, out_size, kernel_size, bias=False)
        self.bn2 = nn.BatchNorm1d(out_size)
        self.downsample = nn.Sequential(
                nn.Conv1d(out_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_size)
            )

    def forward(self, x):
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)

        # downsample
        out = self.downsample(out)
        out = self.relu(out)

        return out


class EmbResNet34_01(nn.Module):

    def __init__(self, input_size, output_size):
        
        super(EmbResNet34_01, self).__init__()

        # Embedding layer
        emb_channel1 = 256
        kernel_size = 3
        self.embedding1 = BasicBlock(input_size, emb_channel1, kernel_size)

        # RNN layer
        num_layers = 4
        hidden_dim = 256
        dropout_rnn = 0.2
        bidirectional = True
        self.lstm = nn.LSTM(
            emb_channel1, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout_rnn,
            bidirectional=bidirectional)
        
        # Linear layer
        linear_layers1 = 2048
        self.linear1 = nn.Linear(hidden_dim*2, linear_layers1)

        dropout = 0.2
        self.dropout = nn.Dropout(dropout)

        # Classification layer
        self.cls_layer = nn.Linear(linear_layers1, output_size)

    def forward(self, x, lx):

        # Embedding layer
        x = torch.transpose(x, 1, 2)
        out = self.embedding1(x)
        out = torch.transpose(out, 1, 2)
        out = self.dropout(out)

        # RNN layer
        # Pack padded the sequence of x to put to LSTM
        shape_diff = out.shape[1] - x.shape[2]
        packed_input = pack_padded_sequence(out, lx+shape_diff, enforce_sorted=False, batch_first=True)
        # RNN
        out, (h_out, h_cell) = self.lstm(packed_input)
        # Unpack the output
        out, lengths = pad_packed_sequence(out, batch_first=True)

        # Linear layer
        out = self.linear1(out)
        out = self.dropout(out)

        # Classification layer
        out = self.cls_layer(out)
        # Log softmax output
        out = F.log_softmax(out, dim=2)

        # Return outputs and length of each output
        return out, lengths

class EmbResNet34_02(nn.Module):

    def __init__(self, input_size, output_size):
        
        super().__init__()

        # Embedding layer
        emb_channel1 = 256
        kernel_size = 3
        self.embedding1 = BasicBlock(input_size, emb_channel1, kernel_size)

        # RNN layer
        num_layers = 4
        hidden_dim = 256
        dropout_rnn = 0.3
        bidirectional = True
        self.lstm = nn.LSTM(
            emb_channel1, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout_rnn,
            bidirectional=bidirectional)
        
        # Linear layer
        linear_layers1 = 2048
        self.linear1 = nn.Linear(hidden_dim*2, linear_layers1)
        self.batchnorm1 = nn.BatchNorm1d(linear_layers1)

        dropout1 = 0.3
        self.dropout1 = nn.Dropout(dropout1)
        dropout2 = 0.4
        self.dropout2 = nn.Dropout(dropout2)
        self.relu = nn.ReLU()

        # Classification layer
        self.cls_layer = nn.Linear(linear_layers1, output_size)

    def forward(self, x, lx):

        # Embedding layer
        x = torch.transpose(x, 1, 2)
        out = self.embedding1(x)
        out = torch.transpose(out, 1, 2)
        out = self.dropout1(out)

        # RNN layer
        # Pack padded the sequence of x to put to LSTM
        shape_diff = out.shape[1] - x.shape[2]
        packed_input = pack_padded_sequence(out, lx+shape_diff, enforce_sorted=False, batch_first=True)
        # RNN
        out, (h_out, h_cell) = self.lstm(packed_input)
        # Unpack the output
        out, lengths = pad_packed_sequence(out, batch_first=True)

        # Linear layer
        out = self.linear1(out)
        out = torch.transpose(out, 1, 2)
        out = self.batchnorm1(out)
        out = torch.transpose(out, 1, 2)
        out = self.dropout2(out)
        out = self.relu(out)

        # Classification layer
        out = self.cls_layer(out)
        # Log softmax output
        out = F.log_softmax(out, dim=2)

        # Return outputs and length of each output
        return out, lengths
