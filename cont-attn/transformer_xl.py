import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()

        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        # NOTE : register buffer = part of the module, will be saved in the state_dict & moved to GPU with the model
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]

class PositionwiseFF(torch.nn.Module):
    def __init__(self, d_input, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_inner), torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_inner, d_input),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input_):
        ff_out = self.ff(input_)
        return ff_out

class GatingMechanism(torch.nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1-z, x) + torch.mul(z, h)
        return g

class MultiHeadAttentionXL(torch.nn.Module):
    def __init__(self, d_input, d_inner, n_heads=4, dropout=0.1, dropouta=0.0):
        super(MultiHeadAttentionXL, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads

        # Linear transformation for keys & values for all heads at once for efficiency
        self.linear_kv = torch.nn.Linear(d_input, (d_inner * n_heads * 2), # 2 for keys & values
            bias=False)
        # for queries (will not be concatenated with memorized states so separate)
        self.linear_q = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)
        
        # for positional embeddings
        self.linear_p = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)
        self.scale = 1 / (d_inner ** 0.5) # for scaled dot product attention
        self.dropa = torch.nn.Dropout(dropouta)

        self.lout = torch.nn.Linear(self.d_inner * self.n_heads, self.d_input, bias=False)
        self.dropo = torch.nn.Dropout(dropout)
        
    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, * x.size()[2:]), device=x.device, dtype=x.dtype)
        return (torch.cat([zero_pad, x], dim=1)
                    .view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:]
                    .view_as(x)) 
        
    def forward(self, input_, pos_embs, memory, u, v, mask=None):
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments  
            - input: torch.FloatTensor, shape - (seq, bs, self.d_input)
            - pos_embs: torch.FloatTensor, shape - (seq + prev_seq, bs, self.d_input)
            - memory: torch.FloatTensor, shape - (prev_seq, b, d_in)
            - u: torch.FloatTensor, shape - (No. of heads, inner_dim)
            - v: torch.FloatTensor, shape - (No. of heads, inner_dim)
            - mask: torch.FloatTensor, Optional 

        + Returns
            - output: torch.FloatTensor, shape - (seq, bs, self.d_input)

        + symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        cur_seq = input_.shape[0] 
        prev_seq = memory.shape[0] 
        H, d = self.n_heads, self.d_inner
        input_with_memory = torch.cat([memory, input_], dim=0) # concat memory across sequence dimension

        k_tfmd, v_tfmd = torch.chunk(self.linear_kv(input_with_memory), 2, dim=-1) # (cs + ps, b, H * d)
        q_tfmd = self.linear_q(input_) # (cs, b, H * d)
        
        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1]

        # i = no. of queries = no. of current inputs/targets (seq-wise)
        # j = no. of key/values
        content_attn = torch.einsum("ibhd,jbhd->ijbh", (
                (q_tfmd.view(cur_seq, bs, H, d) + # (a)
                 u), # (c): u represents the global (independent of query) bias towards certain key/values
                 k_tfmd.view(cur_seq + prev_seq, bs, H, d) # There is no positional information to be found here
        )) # (cs, cs + ps, b, H)
        
        # position-based attention term ((b) + (d) in the paper)
        # this attention is solely based on the position of the key/values
        # (i.e. it does not take the content of the key/values into account)
        p_tfmd = self.linear_p(pos_embs) # (cs + ps, b, H * d)
        position_attn = torch.einsum("ibhd,jhd->ijbh", (
                (q_tfmd.view(cur_seq, bs, H, d) + # (b)
                 v), # (d): v represents the global (independent of the query)
                     # bias towards certain positions
                 p_tfmd.view(cur_seq + prev_seq, H, d) # Notice there is not content information
                                                        # regarding keys and values here!
        )) # (cs, cs + ps, b, H)
        
        #  Compute positional attention efficiently
        position_attn = self._rel_shift(position_attn)
        
        # the attention is the sum of content-based and position-based attention
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            attn = attn.masked_fill(
                mask[..., None], -float('inf'))
        attn = torch.softmax(attn * self.scale, # rescale to prevent values from exploding
                             dim=1) # normalize across the value sequence dimension
        attn = self.dropa(attn)
        
        attn_weighted_values = (torch.einsum("ijbh,jbhd->ibhd",
                                           (attn, # (cs, cs + ps, b, H)
                                            v_tfmd.view(cur_seq + prev_seq, bs, H, d), # (cs + ps, b, H, d)
                                           )) # (cs, b, H, d)
                                .contiguous() # we need to change the memory layout to make `view` work
                                .view(cur_seq, bs, H * d)) # (cs, b, H * d)

        output = self.dropo(self.lout(attn_weighted_values))
        return output


class StableTransformerEncoderLayerXL(torch.nn.Module):
    def __init__(self, n_heads, d_input,  d_head_inner, d_ff_inner, dropout, gating=True, dropouta=0.0):
        super(StableTransformerEncoderLayerXL, self).__init__()

        self.gating = gating
        self.gate1 = GatingMechanism(d_input)
        self.gate2 = GatingMechanism(d_input)
        self.mha = MultiHeadAttentionXL(d_input, d_head_inner, n_heads=n_heads, dropout=dropout, dropouta=dropouta)
        self.ff = PositionwiseFF(d_input, d_ff_inner, dropout)
        self.norm1 = torch.nn.LayerNorm(d_input)
        self.norm2 = torch.nn.LayerNorm(d_input)
            
    def forward(self, input_, pos_embs, u, v, mask=None, mems=None):
        src2 = self.norm1(input_)
        src2 = self.mha(src2, pos_embs, mems, u, v, mask=mask)
        src = self.gate1(input_, src2) if self.gating else input_ + src2
        src2 = self.ff(self.norm2(src))
        src = self.gate2(src, src2) if self.gating else src + src2
        return src

class StableTransformerXL(torch.nn.Module):
    def __init__(self, d_input, n_layers, n_heads, d_head_inner, d_ff_inner,
                 dropout=0.1, dropouta=0.):
        super(StableTransformerXL, self).__init__()

        self.n_layers, self.n_heads, self.d_input, self.d_head_inner, self.d_ff_inner = \
            n_layers, n_heads, d_input, d_head_inner, d_ff_inner

        self.pos_embs = PositionalEmbedding(d_input)
        self.drop = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList([StableTransformerEncoderLayerXL(n_heads, d_input, d_head_inner=d_head_inner,
                                                  d_ff_inner=d_ff_inner, dropout=dropout, dropouta=dropouta)
                                     for _ in range(n_layers)])
        
        # u and v are global parameters: maybe changing these to per-head parameters might help performance?
        self.u, self.v = (torch.nn.Parameter(torch.Tensor(self.n_heads, self.d_head_inner)),
                          torch.nn.Parameter(torch.Tensor(self.n_heads, self.d_head_inner)))
        
    def init_memory(self, device=torch.device("cpu")):
        return [torch.empty(0, dtype=torch.float).to(device) for _ in range(self.n_layers+1)]
    
    def update_memory(self, previous_memory, hidden_states):
        '''
            + Arguments
                - previous_memory: List[torch.FloatTensor],
                - hidden_states: List[torch.FloatTensor]
        '''
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)

        # For the updated memory, we use the most recent `self.mem_len`
        # states, including the previous memory
        # In other words, if `seq_len` < `self.mem_len` some of the previous memory
        # will carry over to the next memory
        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len
            beg_idx = max(0, end_idx - mem_len)
            for m, h in zip(previous_memory, hidden_states):
                cat = torch.cat([m, h], dim=0) # (mem_len + seq_len, bs, d)
                new_memory.append(cat[beg_idx:end_idx].detach()) # (self.mem_len, bs, d)
        return new_memory
    
    def forward(self, inputs, memory=None, mask=None):
        '''
            + Arguments 
                - inputs - torch.FloatTensor
                - memory - Optional, list[torch.FloatTensor]
        '''
        if memory is None: 
            memory = self.init_memory(inputs.device)
        assert len(memory) == len(self.layers) + 1
        
        cur_seq, bs = inputs.shape[:2]
        prev_seq = memory[0].size(0)
        
        # TODO - maybe take mask as bool and if true, pass this instead of custom mask?
        # dec_attn_mask = torch.triu(
        #     torch.ones((cur_seq, cur_seq + prev_seq)),
        #     diagonal=1 + prev_seq,
        # ).byte()[..., None].to(inputs.device)
        
        pos_ips = torch.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=torch.float).to(inputs.device)
        pos_embs = self.drop(self.pos_embs(pos_ips))
        if self.d_input % 2 != 0:
            pos_embs = pos_embs[:, :, :-1]
        
        hidden_states = [inputs]
        layer_out = inputs
        for mem, layer in zip(memory, self.layers):
            layer_out = layer(layer_out, pos_embs, self.u, self.v, 
                              mask=mask, mems=mem)
            hidden_states.append(layer_out)
        
        # Update memory - Memory is treated as a const., we don't back propagate through it
        new_memory = self.update_memory(memory, hidden_states)
        return {"logits": layer_out, "memory": new_memory}

if __name__ == '__main__':
    states = torch.randn(20, 5, 8) # seq_size, batch_size, dim - better if dim % 2 == 0
    # print("=> Testing Transformer")
    # transformer = StableTransformerXL(d_input=states.shape[-1], n_layers=4, n_heads=3, d_head_inner=32, d_ff_inner=71)
    # output = transformer(states, memory=None)
    # output, mem = output['logits'], output['memory']
    # print(output.shape, len(mem), mem[0].shape, torch.isinf(output).any(), torch.isnan(output).any())
    # output = transformer(states, memory=mem)
    # output, mem = output['logits'], output['memory']
    # print(output.shape, len(mem), mem[0].shape, torch.isinf(output).any(), torch.isnan(output).any())

    print("=> Testing Policy")
    policy = TransformerGaussianPolicy(state_dim=states.shape[-1], act_dim=4)
    act = policy(states)
    action = act[0].sample()
    print(torch.isnan(action).any(), action.shape)