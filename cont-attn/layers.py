import copy
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

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

class RLTransofmerEncoderLayer(torch.nn.Module):
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

    def __init__(self, state_dim, action_dim, nhead=8, gating=False, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(RLTransofmerEncoderLayer, self).__init__()

        self.state_action_attn = RLMultiHeadAttention(state_dim, action_dim, nhead)
        self.linear1 = torch.nn.Linear(state_dim, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, action_dim)

        self.norm1 = torch.nn.LayerNorm(state_dim)
        self.norm2 = torch.nn.LayerNorm(action_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, curr_state, state_set, action_set, attn_mask=None, key_padding_mask=None):
        # NOTE - in the paper, LayerNorm is applied before first layer too.
        src2 = self.state_action_attn(curr_state, state_set, action_set, attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class StableTransformerEncoderLayer(torch.nn.Module):
    '''
        + Transformer(not-XL) implemented from https://arxiv.org/abs/1910.06764.
        + Self-Attention for state-space, model same as above.
    '''
    def __init__(self, state_dim, nhead=8, gating=False, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(StableTransformerEncoderLayer, self).__init__()

        self.self_attn = torch.nn.MultiheadAttention(embed_dim=state_dim, num_heads=nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(state_dim, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, state_dim)

        self.norm1 = torch.nn.LayerNorm(state_dim)
        self.norm2 = torch.nn.LayerNorm(state_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, state_set, attn_mask=None, key_padding_mask=None):
        # NOTE - in the paper, LayerNorm is applied before first layer too.
        src2 = self.norm1(state_set)
        src2 = self.self_attn(src2, src2, src2, attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask)
        src = state_set + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src