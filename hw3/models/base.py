# Starter code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Network01(nn.Module):

    def __init__(self, input_size, output_size):
        
        super(Network01, self).__init__()
        # Simple network with 1 LSTM and one Lienar layer
        hidden_dim = 256
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True) 
        self.classification = nn.Linear(hidden_dim, output_size)

    def forward(self, x, lx):

        # Pack padded the sequence of x to put to LSTM
        packed_input = pack_padded_sequence(x, lx, enforce_sorted=False, batch_first=True)

        # RNN layer
        out1, (h_out, h_cell) = self.lstm(packed_input)

        # Unpack the output
        out, lengths  = pad_packed_sequence(out1, batch_first=True)

        # Classify layer
        out = self.classification(out)
        # Log softmax output
        out = F.log_softmax(out, dim=2)

        # Return outputs and length of each output
        return out, lengths


class Network02(nn.Module):

    def __init__(self, input_size, output_size):
        
        super(Network02, self).__init__()

        # model params
        out_channel = 64
        kernel_size = 3
        hidden_dim = 256

        self.embedding = nn.Conv1d(input_size, out_channel, kernel_size, stride=1)
        self.lstm = nn.LSTM(out_channel, hidden_dim, batch_first=True)
        self.classification = nn.Linear(hidden_dim, output_size)

    def forward(self, x, lx):

        # Embedding layer
        out_emb = self.embedding(torch.transpose(x, 1, 2))
        out_emb = torch.transpose(out_emb, 1, 2)

        # Pack padded the sequence of x to put to LSTM
        shape_diff = out_emb.shape[1] - x.shape[1]
        packed_input = pack_padded_sequence(out_emb, lx+shape_diff, enforce_sorted=False, batch_first=True)

        # RNN layer
        out_rnn, (h_out, h_cell) = self.lstm(packed_input)

        # Unpack the output
        out_rnn, lengths  = pad_packed_sequence(out_rnn, batch_first=True)

        # Classify layer
        out = self.classification(out_rnn)
        # Log softmax output
        out = F.log_softmax(out, dim=2)

        # Return outputs and length of each output
        return out, lengths


class Network03(nn.Module):

    def __init__(self, input_size, output_size):
        
        super(Network03, self).__init__()

        emb_channel1 = 64
        emb_channel2 = 256
        kernel_size = 3
        self.embedding1 = nn.Conv1d(input_size, emb_channel1, kernel_size, stride=1)
        self.embedding2 = nn.Conv1d(emb_channel1, emb_channel2, kernel_size, stride=1)

        num_layers = 4
        hidden_dim = 256
        dropout = 0.1
        bidirectional = True
        self.lstm = nn.LSTM(
            emb_channel2, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout,
            bidirectional=bidirectional)
        
        linear_layers1 = 2048
        self.linear1 = nn.Linear(hidden_dim*2, linear_layers1)
        self.cls_layer = nn.Linear(linear_layers1, output_size)

    def forward(self, x, lx):

        # Embedding layer
        out = self.embedding1(torch.transpose(x, 1, 2))
        out = self.embedding2(out)
        out = torch.transpose(out, 1, 2)

        # RNN layer
        # Pack padded the sequence of x to put to LSTM
        shape_diff = out.shape[1] - x.shape[1]
        packed_input = pack_padded_sequence(out, lx+shape_diff, enforce_sorted=False, batch_first=True)
        # RNN
        out, (h_out, h_cell) = self.lstm(packed_input)
        # Unpack the output
        out, lengths  = pad_packed_sequence(out, batch_first=True)

        # Linear layer
        out = self.linear1(out)
        out = self.cls_layer(out)
        # Log softmax output
        out = F.log_softmax(out, dim=2)

        # Return outputs and length of each output
        return out, lengths


class Network04(nn.Module):

    def __init__(self, input_size, output_size):
        
        super(Network04, self).__init__()

        # Embedding layer
        emb_channel1 = 64
        emb_channel2 = 256
        kernel_size = 3
        self.embedding1 = nn.Conv1d(input_size, emb_channel1, kernel_size, stride=1)
        self.embedding2 = nn.Conv1d(emb_channel1, emb_channel2, kernel_size, stride=1)

        # RNN layer
        num_layers = 4
        hidden_dim = 256
        dropout_rnn = 0.1
        bidirectional = True
        self.lstm = nn.LSTM(
            emb_channel2, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout_rnn,
            bidirectional=bidirectional)
        
        # Linear layer
        linear_layers1 = 2048
        self.linear1 = nn.Linear(hidden_dim*2, linear_layers1)
        dropout_linear = 0.1
        self.dropout1 = nn.Dropout(dropout_linear)

        # Classification layer
        self.cls_layer = nn.Linear(linear_layers1, output_size)

    def forward(self, x, lx):

        # Embedding layer
        out = self.embedding1(torch.transpose(x, 1, 2))
        out = self.embedding2(out)
        out = torch.transpose(out, 1, 2)

        # RNN layer
        # Pack padded the sequence of x to put to LSTM
        shape_diff = out.shape[1] - x.shape[1]
        packed_input = pack_padded_sequence(out, lx+shape_diff, enforce_sorted=False, batch_first=True)
        # RNN
        out, (h_out, h_cell) = self.lstm(packed_input)
        # Unpack the output
        out, lengths  = pad_packed_sequence(out, batch_first=True)

        # Linear layer
        out = self.linear1(out)
        out = self.dropout1(out)
        out = self.cls_layer(out)
        # Log softmax output
        out = F.log_softmax(out, dim=2)

        # Return outputs and length of each output
        return out, lengths
