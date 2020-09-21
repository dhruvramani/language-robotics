import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

class BasicAttnModel(torch.nn.Module):
    def __init__(self, action_dim, nhead=8, dropout=0.1):
        super(BasicAttnModel, self).__init__()
        self.attn = torch.nn.MultiheadAttention(action_dim, nhead, dropout=dropout)

    def forward(self, curr_state, state_set, action_set):
        attn_outputs, attn_weights = self.attn(query=curr_state, key=state_set, value=action_set)
        return attn_outputs