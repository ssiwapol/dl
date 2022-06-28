import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BasicBlock(nn.Module):

    # input = (batch_size, in_size, T)
    # output = (batch_size, out_size, roundup(T/stride))
    
    def __init__(self, in_size, out_size, kernel_size, stride=1):
        super().__init__()
        # accept only kernel_size = odd no.
        pad_size = int(kernel_size / 2)
        self.stride = stride

        # conv1
        self.conv1 = nn.Conv1d(
            in_size, out_size, kernel_size, stride, 
            bias=False, padding=pad_size)
        self.bn1 = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()

        # conv2
        self.conv2 = nn.Conv1d(
            out_size, out_size, kernel_size, 
            bias=False, padding=pad_size)
        self.bn2 = nn.BatchNorm1d(out_size)

        # downsample
        self.downsample = nn.Sequential(
            nn.Conv1d(
                in_size, out_size, 
                kernel_size=1, stride=stride, bias=False),
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
        identity = self.downsample(x)

        # out + idenity
        out += identity

        # relu
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    # input = (batch_size, in_size, T)
    # output = (batch_size, out_size, roundup(T/stride))

    def __init__(self, in_size, out_size, kernel_size, stride=1):
        super().__init__()
        # accept only kernel_size = odd no.
        pad_size = int(kernel_size / 2)
        self.stride = stride
        self.relu = nn.ReLU()

        # conv1
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_size)

        # conv2
        self.conv2 = nn.Conv1d(out_size, out_size, kernel_size, stride=stride, bias=False, padding=pad_size)
        self.bn2 = nn.BatchNorm1d(out_size)

        # conv3
        self.conv3 = nn.Conv1d(out_size, out_size, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_size)

        # downsample
        self.downsample = nn.Sequential(
            nn.Conv1d(
                in_size, out_size, 
                kernel_size=1, stride=stride, bias=False),
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
        out = self.relu(out)

        # conv3
        out = self.conv3(out)
        out = self.bn3(out)

        # downsample
        identity = self.downsample(x)

        # out + idenity
        out += identity

        # relu
        out = self.relu(out)

        return out


class EmbResNet34_03(nn.Module):

    def __init__(self, input_size, output_size):
        
        super().__init__()

        # Embedding layer
        emb_channel1 = 256
        kernel_size = 3
        self.stride = 1
        self.embedding1 = BasicBlock(input_size, emb_channel1, kernel_size, self.stride)
        # Dropout after embedding
        dropout1 = 0.3
        self.dropout1 = nn.Dropout(dropout1)

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
        # Dropout after linear layer
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
        lx1 = torch.ceil(lx / self.stride)
        packed_input = pack_padded_sequence(out, lx1, enforce_sorted=False, batch_first=True)
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


class EmbResNet34_04(nn.Module):

    def __init__(self, input_size, output_size):
        
        super().__init__()

        # Embedding layer
        emb_channel1 = 256
        kernel_size = 3
        self.stride = 2
        self.embedding1 = BasicBlock(input_size, emb_channel1, kernel_size, self.stride)
        # Dropout after embedding
        dropout1 = 0.3
        self.dropout1 = nn.Dropout(dropout1)

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
        # Dropout after linear layer
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
        lx1 = torch.ceil(lx / self.stride)
        packed_input = pack_padded_sequence(out, lx1, enforce_sorted=False, batch_first=True)
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


class EmbResNet34_05(nn.Module):

    def __init__(self, input_size, output_size):
        
        super().__init__()

        # Embedding layer
        emb_channel1 = 256
        kernel_size = 3
        self.stride = 2
        self.embedding1 = BasicBlock(input_size, emb_channel1, kernel_size, self.stride)
        # Dropout after embedding
        dropout1 = 0.3
        self.dropout1 = nn.Dropout(dropout1)

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
        # Dropout after linear layer
        dropout2 = 0.4
        self.dropout2 = nn.Dropout(dropout2)
        self.relu = nn.ReLU()

        # Classification layer
        self.cls_layer = nn.Linear(linear_layers1, output_size)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, lx):

        # Embedding layer
        x = torch.transpose(x, 1, 2)
        out = self.embedding1(x)
        out = torch.transpose(out, 1, 2)
        out = self.dropout1(out)

        # RNN layer
        # Pack padded the sequence of x to put to LSTM
        lx1 = torch.ceil(lx / self.stride)
        packed_input = pack_padded_sequence(out, lx1, enforce_sorted=False, batch_first=True)
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


class EmbResNet50_01(nn.Module):

    def __init__(self, input_size, output_size):
        
        super().__init__()

        # Embedding layer
        emb_channel1 = 256
        kernel_size = 3
        self.stride = 2
        self.embedding1 = Bottleneck(input_size, emb_channel1, kernel_size, self.stride)
        # Dropout after embedding
        dropout1 = 0.3
        self.dropout1 = nn.Dropout(dropout1)

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
        # Dropout after linear layer
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
        lx1 = torch.ceil(lx / self.stride)
        packed_input = pack_padded_sequence(out, lx1, enforce_sorted=False, batch_first=True)
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


class EmbResNet50_02(nn.Module):

    def __init__(self, input_size, output_size):
        
        super().__init__()

        # Embedding layer
        emb_channel1 = 256
        kernel_size = 3
        self.stride = 2
        self.embedding1 = Bottleneck(input_size, emb_channel1, kernel_size, self.stride)
        # Dropout after embedding
        dropout1 = 0.3
        self.dropout1 = nn.Dropout(dropout1)

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
        # Dropout after linear layer
        dropout2 = 0.4
        self.dropout2 = nn.Dropout(dropout2)
        self.relu = nn.ReLU()

        # Classification layer
        self.cls_layer = nn.Linear(linear_layers1, output_size)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, lx):

        # Embedding layer
        x = torch.transpose(x, 1, 2)
        out = self.embedding1(x)
        out = torch.transpose(out, 1, 2)
        out = self.dropout1(out)

        # RNN layer
        # Pack padded the sequence of x to put to LSTM
        lx1 = torch.ceil(lx / self.stride)
        packed_input = pack_padded_sequence(out, lx1, enforce_sorted=False, batch_first=True)
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


class EmbResNet50_03(nn.Module):

    def __init__(self, input_size, output_size):
        
        super().__init__()

        self.stride = 2
        # Embedding layer
        emb_channel1 = 512
        kernel_size1 = 3
        self.stride1 = 2
        self.embedding1 = Bottleneck(input_size, emb_channel1, kernel_size1, self.stride1)
        emb_channel2 = 256
        kernel_size2 = 3
        self.stride2 = 1
        self.embedding2 = Bottleneck(input_size, emb_channel1, kernel_size2, self.stride2)
        # Dropout after embedding
        dropout1 = 0.3
        self.dropout1 = nn.Dropout(dropout1)

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
        # Dropout after linear layer
        dropout2 = 0.4
        self.dropout2 = nn.Dropout(dropout2)
        self.relu = nn.ReLU()

        # Classification layer
        self.cls_layer = nn.Linear(linear_layers1, output_size)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, lx):

        # Embedding layer
        x = torch.transpose(x, 1, 2)
        out = self.embedding1(x)
        out = self.embedding2(x)
        out = torch.transpose(out, 1, 2)
        out = self.dropout1(out)

        # RNN layer
        # Pack padded the sequence of x to put to LSTM
        lx1 = torch.ceil(torch.ceil(lx / (self.stride1)) / self.stride2)
        packed_input = pack_padded_sequence(out, lx1, enforce_sorted=False, batch_first=True)
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
