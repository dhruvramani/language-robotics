import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

class RLTransofmerEncoderLayer(torch.nn.Module):
    '''
        + Transformer(not-XL) implemented from https://arxiv.org/abs/1910.06764.
            - Key, Queries : States
            - Values : Actions corresponding to the Key.

        + Args:
            - action_dim: the number of expected features in the action (value) (required).
            - nhead: the number of heads in the multiheadattention models (required).
            - dim_feedforward: the dimension of the feedforward network model (default=2048).
            - dropout: the dropout value (default=0.1).
            - activation: the activation function of intermediate layer, relu or gelu (default=relu)

    '''

    def __init__(self, action_dim, nhead, gating=False, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(RLTransofmerEncoderLayer, self).__init__()

        self.self_attn = torch.nn.MultiheadAttention(action_dim, nhead)
        self.linear1 = Linear(action_dim, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, action_dim)
        # TODO : Complete this
