import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

import layers

class BasicAttnModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BasicAttnModel, self).__init__()

        self.attn = layers.RLMultiHeadAttention(state_dim, action_dim)
        self.projection = torch.nn.Linear(state_dim, action_dim)

    def forward(self, curr_state, state_set, action_set):
        # TODO - IMPORTANT : See if attention is being applied over batch - wrong if only over seqs.
        # If seqs, reshape to have batch_size=1
        output = self.attn(curr_state, state_set, action_set)
        output = self.projection(output)
        return output

class RLTransformerEncoder(torch.nn.Module):
    '''
        Passes the state-vectors through number of self-attention layers.
        Then applies RLTransofmerEncoderLayer over the state and action vectors.
    '''
    def __init__(self, state_dim, action_dim, n_state_encoders):
        super(RLTransformerEncoder, self).__init__()

        state_encoder = layers.StableTransformerEncoderLayer(state_dim)
        self.layers = layers._get_clones(state_encoders, n_state_encoders)
        self.final_layer = layers.RLTransofmerEncoderLayer(state_dim, action_dim)

    def forward(self, curr_state, state_set, action_set, mask=None, state_key_padding_mask=None):
        for mod in self.layers:
            state_set = mod(state_set, attn_mask=mask, key_padding_mask=state_key_padding_mask)
            curr_state = mod(curr_state, attn_mask=mask, key_padding_mask=state_key_padding_mask)

        pred_action = self.final_layer(curr_state, state_set, action_set, attn_mask=mask, key_padding_mask=state_key_padding_mask)
        return pred_action