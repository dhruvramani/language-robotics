import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

import transformer_xl as txl

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class SpatialSoftmax(torch.nn.Module):
    ''' 
        Spatial softmax is used to find the expected pixel location of feature maps.
        Source : https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834 
        Output : Tensor of shape (N, C * 2) - don't use flatten!
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
        pos_x = torch.FloatTensor(pos_x.reshape(self.height * self.width))
        pos_y = torch.FloatTensor(pos_y.reshape(self.height * self.width))
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

class RLMultiHeadAttention(torch.nn.Module):
    ''' NOTE : embed_dim = query's dim, O/P dim = embed_dim '''
    def __init__(self, state_dim, act_dim, nhead=8, dropout=0.1):
        super(RLMultiHeadAttention, self).__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=state_dim, kdim=state_dim, vdim=act_dim, 
            num_heads=nhead, dropout=dropout)
        

    def forward(self, curr_state, state_set, action_set, attn_mask=None, key_padding_mask=None):
        output, attn_weights = self.attn(query=curr_state, key=state_set, value=action_set, attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask)
        return output

class RLTransorfmerEncoderLayer(torch.nn.Module):
    '''
        + Transformer(not-XL) implemented from https://arxiv.org/abs/1910.06764.
            - Key, Queries : States
            - Values : Actions corresponding to the Key.

        + Args:
            - state_dim: the number of expected features in the state (value) (required).
            - action_dim: the number of expected features in the action (value) (required).
            - nhead: the number of heads in the multiheadattention models (required).
            - dim_feedforward: the dimension of the feedforward network model (default=2048).
            - dropout: the dropout value (default=0.1).
            - activation: the activation function of intermediate layer, relu or gelu (default=gelu)

    '''
    def __init__(self, state_dim, action_dim, n_heads, dropout=0.1, gating=True):
        super(RLTransorfmerEncoderLayer, self).__init__()

        self.gating = gating
        self.gate1 = txl.GatingMechanism(state_dim)
        self.gate2 = txl.GatingMechanism(state_dim)
        
        self.mha = RLMultiHeadAttention(state_dim, action_dim, nhead=n_heads, dropout=dropout)
        self.ff = txl.PositionwiseFF(state_dim, d_inner=64, dropout=dropout)
        self.lin1 = torch.nn.Linear(state_dim, action_dim)

        self.norm1 = torch.nn.LayerNorm(state_dim)
        self.norm2 = torch.nn.LayerNorm(state_dim)
            
    def forward(self, curr_state, state_set, action_set, mask=None):
        src2 = self.norm1(curr_state)
        src2 = self.mha(curr_state, state_set, action_set, key_padding_mask=mask)
        src = self.gate1(curr_state, src2) if self.gating else input_ + src2
        src2 = self.ff(self.norm2(src))
        src = self.gate2(src, src2) if self.gating else src + src2
        return src