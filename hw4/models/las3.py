import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import letter2index


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5, batch_first=True):
        '''
        https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html
        batch_fist=True: [batch_size, sequence_length, rnn_hidden_size]
        batch_fist=False: [sequence_length, batch_size, rnn_hidden_size]
        '''
        if not self.training or not dropout:
            return x
        x = x.clone()
        if batch_first:
            mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - dropout)
        else:
            mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - dropout)
        mask = mask.div_(1 - dropout)
        mask = mask.expand_as(x)
        return x * mask

class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    '''
    def __init__(self, input_dim, hidden_dim, trunc='concat'):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1,
                             bidirectional=True, batch_first=True)
        self.trunc = trunc

    def forward(self, x, lx):
        # Get x from lstm output
        # Pad packed first
        x = x[:, :(x.shape[1] // 2) * 2, :]
        #lx = torch.div(lx, 2, rounding_mode='floor')  # torch >= 1.8
        lx = torch.floor_divide(lx, 2)   # torch 1.7.1
        # Truncate input length
        if self.trunc == 'mean':
            x = x.view(x.shape[0], x.shape[1] // 2, 2, x.shape[2])
            x = torch.mean(x, dim=2)
        elif self.trunc == 'max':
            x = x.view(x.shape[0], x.shape[1] // 2, 2, x.shape[2])
            x = torch.amax(x, dim=2)
        else:
            x = x.view(x.shape[0], x.shape[1] // 2, x.shape[2] * 2)
        # Pack padded sequence
        x_pack = pack_padded_sequence(x, lx, batch_first=True, enforce_sorted=False)
        # Pass to model
        out, h = self.blstm(x_pack)
        out, lout = pad_packed_sequence(out, batch_first=True)
        return out, lout

class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self, input_dim, encoder_hidden_dim, 
                 encoder_lockdrop=None, encoder_dropout=0.1,
                 trunc='mean', key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM layer at the bottom
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_hidden_dim, num_layers=1, 
                            bidirectional=True, batch_first=True)

        # Define the blocks of pBLSTMs
        if trunc=='concat':
            input_pblstm = encoder_hidden_dim * 4
        else:
            input_pblstm = encoder_hidden_dim * 2
        # Construct 3 layers of pBLSTM
        self.pBLSTMs = nn.ModuleList([
            pBLSTM(input_pblstm, encoder_hidden_dim, trunc),
            pBLSTM(input_pblstm, encoder_hidden_dim, trunc),
            pBLSTM(input_pblstm, encoder_hidden_dim, trunc)
        ])

        self.lockdrop = LockedDropout()
        self.encoder_lockdrop = encoder_lockdrop
        self.dropout = nn.Dropout(encoder_dropout)
         
        # The linear transformations for producing Key and Value for attention
        self.key_network = nn.Linear(encoder_hidden_dim*2, key_value_size)
        self.value_network = nn.Linear(encoder_hidden_dim*2, key_value_size)

    def forward(self, x, x_len):
        x_pack = pack_padded_sequence(x, x_len, enforce_sorted=False, batch_first=True)
        out, h = self.lstm(x_pack)
        out, lout = pad_packed_sequence(out, batch_first=True)
        for i, rnn in enumerate(self.pBLSTMs):
            out, lout = rnn(out, lout)
            # Implement locked drop between pBLSTM layers
            if self.encoder_lockdrop:
                if i < len(self.pBLSTMs)-1:
                    out = self.lockdrop(out, self.encoder_lockdrop)
        out = self.dropout(out)
        key = self.key_network(out)
        val = self.value_network(out)
        return key, val, lout

class Attention(nn.Module):
    '''
    Attention is calculated using key and value from encoder and query from decoder.
    '''
    def __init__(self, dropout=None):
        super(Attention, self).__init__()
        # Optional: dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, query, key, value, mask):
        # Calculate (Q dot K^T) / dk
        dk = query.shape[-1]
        energy = torch.bmm(query.unsqueeze(1), key.transpose(-2, -1)) / np.sqrt(dk)  # (B, 1, seq)
        # Add mask
        #energy_m = energy.masked_fill(mask.unsqueeze(1)==0, -1e9)
        energy_m = energy.masked_fill(mask.unsqueeze(1)==0, float('-inf'))
        # Apply softmax
        attn = F.softmax(energy_m, dim = -1).squeeze(1)
        if self.dropout:
            attn = self.dropout(attn)
        # dot V
        context = torch.bmm(attn.unsqueeze(1), value).squeeze(1)
        return context, attn

class Decoder(nn.Module):
    '''
    Decoder: each forward call of decoder deals with just one time step
    '''
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, 
                 device, attn_dropout=None, key_value_size=128):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # The number of cells is defined based on the paper
        self.lstm1 = nn.LSTMCell(embed_dim+key_value_size, decoder_hidden_dim)  # input=cat(emb, context)
        self.lstm2 = nn.LSTMCell(decoder_hidden_dim, key_value_size)  # input=hidden1
    
        self.attention = Attention(attn_dropout)
        self.vocab_size = vocab_size
        # Optional: Weight-tying
        self.character_prob = nn.Linear(key_value_size*2, vocab_size) #: d_v -> vocab_size  input=cat(query, context)
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight

        self.device = device

    def forward(self, key, value, encoder_len, y=None, mode='train', epoch=None):
        B, key_seq_max_len, key_value_size = key.shape

        if mode == 'train':
            max_len =  y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 600

        # Adjust param by epoch
        if epoch:
            # Decrease teacher forcing rate 0.1 every 5 epochs until 0.7
            teacher_forcing_rate = max(1 - (epoch // 5) * 0.1, 0.7)
        else:
            teacher_forcing_rate = 1

        mask = torch.zeros(B, key_seq_max_len)
        for i in range(B):
            mask[i][:encoder_len[i]] = 1
        mask = mask.to(self.device)
        
        predictions = []
        prediction = torch.full((B,), fill_value=letter2index['<sos>'], device=self.device)
        hidden_states = [None, None] 
        
        query = torch.zeros(B, key_value_size, device=self.device)
        context, attention = self.attention(query, key, value, mask)

        #context = value[:, 0]

        attention_plot = [] # this is for debugging

        for i in range(max_len):
            if mode == 'train':
                # Implement Teacher Forcing  
                teacher_forcing = True if np.random.rand() <= teacher_forcing_rate else False
                if i == 0:
                        char_embed = self.embedding(prediction)
                else:
                    if teacher_forcing:
                        char_embed = char_embeddings[:, i-1]
                    else:
                        char_embed = self.embedding(prediction.argmax(dim=-1))
            else:
                if i==0:
                    char_embed = self.embedding(prediction) # embedding of the previous prediction
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))

            y_context = torch.cat([char_embed, context], dim=1)
            # context and hidden states of lstm 1 from the previous time step should be fed
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])

            # hidden states of lstm1 and hidden states of lstm2 from the previous time step should be fed
            hidden1 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(hidden1, hidden_states[1])

            query = hidden_states[1][0]
            
            # Compute attention from the output of the second LSTM Cell
            context, attention = self.attention(query, key, value, mask)
            # We store the first attention of this batch for debugging
            attention_plot.append(attention[0].detach().cpu())
            
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            # store predictions
            predictions.append(prediction.unsqueeze(1))
        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        return predictions, attentions

class Las04(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(Las04,self).__init__()

        # Model parameters
        encoder_hidden_dim=256
        encoder_lockdrop = 0.2
        encoder_dropout=0.2
        decoder_hidden_dim=512
        embed_dim=256
        attn_dropout=0.2
        trunc='mean'
        key_value_size=128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(input_dim, encoder_hidden_dim, device, 
                               encoder_lockdrop, encoder_dropout, 
                               trunc, key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, device,
                               attn_dropout, key_value_size)
        

    def forward(self, x, x_len, y=None, mode='train', epoch=None):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, 
                                               mode=mode, epoch=epoch)
        return predictions, attentions

class Las05(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(Las05,self).__init__()

        # Model parameters
        encoder_hidden_dim=256
        encoder_lockdrop = 0.2
        encoder_dropout=0.2
        decoder_hidden_dim=512
        embed_dim=256
        attn_dropout=0.2
        trunc='concat'
        key_value_size=128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(input_dim, encoder_hidden_dim, 
                               encoder_lockdrop, encoder_dropout, 
                               trunc, key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, device,
                               attn_dropout, key_value_size)
        

    def forward(self, x, x_len, y=None, mode='train', epoch=None):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, 
                                               mode=mode, epoch=epoch)
        return predictions, attentions

class Las06(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(Las06,self).__init__()

        # Model parameters
        encoder_hidden_dim=256
        encoder_lockdrop = 0.2
        encoder_dropout=0.2
        decoder_hidden_dim=512
        embed_dim=256
        attn_dropout=0.2
        trunc='max'
        key_value_size=128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(input_dim, encoder_hidden_dim, 
                               encoder_lockdrop, encoder_dropout, 
                               trunc, key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, device,
                               attn_dropout, key_value_size)
        

    def forward(self, x, x_len, y=None, mode='train', epoch=None):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, 
                                               mode=mode, epoch=epoch)
        return predictions, attentions
