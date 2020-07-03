import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class SpatialSoftmax(torch.nn.Module):
    ''' 
        Spatial softmax is used to find the expected pixel location of feature maps.
        Source : https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834 
    '''
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., self.height), np.linspace(-1., 1., self.width))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints
    
class ConditionalVAE(torch.nn.Module):
    '''
        CoditionalVAE : z ~ p(z | x, c)
        + input_size  : combined dimension of x, c.
        + layers_size : list of dims for diff layers of the encoder.
    '''
    def __init__(self, input_size, latent_size=256, layers_size=[2048] * 4, decoder=False):
        super(self, ConditionalVAE).__init__()
        self.input_size = self.input_size
        self.latent_size = self.latent_size
        self.layers_size = self.layers_size

        assert type(self.layers_size) == list

        self.decoder = decoder

        self.encoder_network = torch.nn.Sequential()
        self.encoder_network.add_module(torch.nn.Linear(self.input_size, self.layers_size[0]))
        self.encoder_network.add_module(torch.nn.ReLU())

        for i, (in_size, out_size) in enumerate(self.layer_sizes[1:]):
            self.encoder_network.add_module(torch.nn.Linear(in_size, out_size))
            self.encoder_network.add_module(torch.nn.ReLU())

        self.hidden2mean = torch.nn.Linear(self.layer_sizes[-1], self.latent_size)
        self.hidden2logv = torch.nn.Linear(self.layer_sizes[-1], self.latent_size) # TODO : Maybe keep logstd fixed
        self.softplus = torch.nn.Softplus() # TODO : Check w/ Softplus

        if self.decoder:
            self.dlayers_size = self.layers_size.reverse()
            self.decoder_network = torch.nn.Sequential()
            self.decoder_network.add_module(torch.nn.Linear(self.latent_size, self.dlayers_size[0]))
            self.decoder_network.add_module(torch.nn.ReLU())

            for i, (in_size, out_size) in enumerate(self.dlayer_sizes[1:]):
                self.decoder_network.add_module(torch.nn.Linear(in_size, out_size))
                self.decoder_network.add_module(torch.nn.ReLU())

            self.decoder_network.add_module(torch.nn.Linear(self.dlayers_size[0], self.input_size))            

    def forward(self, x, c):
        assert x.size()[-1] + c.size()[-1] == self.input_size
        
        x = torch.cat((x, c), dim=-1)
        x = self.encoder_network(x)
        
        mean = self.hidden2mean(x)
        logv = self.hidden2logv(x)
        std = torch.exp(0.5 * logv)
        
        z = torch.randn([batch_size, self.latent_size])
        z = z * std + mean

        if self.decoder:
            recon_x = self.decoder_network(z)
            return recon_x, z, mean, logv

        return z, mean, logv

class SeqVAE(torch.nn.Module):
    '''
        Input sequence shape : (batch, seq, feature)
    '''
    def __init__(self, max_sequence_length, input_size, latent_size=256, hidden_size=2048, rnn_type='RNN', decoder=False, num_layers=2, bidirectional=True):
        super(self, ConditionalSeqVAE).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_sequence_length = max_sequence_length
        
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.decoder = decoder
        self.bidirectional = bidirectional

        assert self.rnn_type in ['LSTM', 'GRU', 'RNN']

        rnn = {'LSTM' : torch.nn.LSTM, 'RNN' : torch.nn.RNN, 'GRU' : torch.nn.GRU}[self.rnn_type]
        self.encoder_rnn = rnn(self.input_size, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)

        if self.decoder:
            self.decoder_rnn = rnn(self.hidden_size, self.input_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.hidden2mean = torch.nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        self.hidden2logv = torch.nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        self.softplus = torch.nn.Softplus() # TODO : Check w/ Softplus

        if self.decoder:
            self.latent2hidden = torch.nn.Linear(self.latent_size, self.hidden_size * self.hidden_factor)

    def forward(self, input_sequence):        
        batch_size = input_sequence.size(0)
        #sorted_lengths, sorted_idx = torch.sort(self.max_sequence_length, descending=True)
        #input_sequence = input_sequence[sorted_idx]
        #input_sequence = torch.nn.rnn_utils.pack_padded_sequence(input_sequence, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(input_sequence)
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        if self.decoder :
            hidden = self.latent2hidden(z)

            if self.bidirectional or self.num_layers > 1:
                hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
            else:
                hidden = hidden.unsqueeze(0)

            outputs, _ = self.decoder_rnn(packed_input, hidden)
            return outputs, z, mean, logv

        return z, mean, logv
