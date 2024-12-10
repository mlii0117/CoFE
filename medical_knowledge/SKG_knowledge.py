import copy
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def sample_next_word(logprobs, sample_method, temperature):
    if sample_method == 'greedy':
        sampleLogprobs, it = torch.max(logprobs.data, 1)
        it = it.view(-1).long()
    elif sample_method == 'gumbel':  # gumbel softmax
        def sample_gumbel(shape, eps=1e-20):
            U = torch.rand(shape).cuda()
            return -torch.log(-torch.log(U + eps) + eps)

        def gumbel_softmax_sample(logits, temperature):
            y = logits + sample_gumbel(logits.size())
            return F.log_softmax(y / temperature, dim=-1)

        _logprobs = gumbel_softmax_sample(logprobs, temperature)
        _, it = torch.max(_logprobs.data, 1)
        sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
    else:
        logprobs = logprobs / temperature
        if sample_method.startswith('top'):  # topk sampling
            top_num = float(sample_method[3:])
            if 0 < top_num < 1:
                # nucleus sampling from # The Curious Case of Neural Text Degeneration
                probs = F.softmax(logprobs, dim=1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                _cumsum = sorted_probs.cumsum(1)
                mask = _cumsum < top_num
                mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
                sorted_probs = sorted_probs * mask.float()
                sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                logprobs.scatter_(1, sorted_indices, sorted_probs.log())
            else:
                the_k = int(top_num)
                tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                topk, indices = torch.topk(logprobs, the_k, dim=1)
                tmp = tmp.scatter(1, indices, topk)
                logprobs = tmp
        it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
        sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
    return it, sampleLogprobs


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states):
        for layer in self.layers:
            x, y = layer(x, hidden_states)
        return self.norm(x), y


class DecoderLayer(nn.Module):
    def __init__(self, d_model, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        # self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, hidden_states):
        m = hidden_states
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        # x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        # return self.sublayer[2](x, self.feed_forward), self.self_attn.attn, self.src_attn.attn
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[1](x, self.feed_forward), self.src_attn.attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):

    def __init__(self, input_dim, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        # self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())
        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

        self.proj1 = nn.Linear(input_dim, self.d_model)
    # def init_memory(self, batch_size):
    #     memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
    #     if self.d_model > self.num_slots:
    #         diff = self.d_model - self.num_slots
    #         pad = torch.zeros((batch_size, self.num_slots, diff))
    #         memory = torch.cat([memory, pad], -1)
    #     elif self.d_model < self.num_slots:
    #         memory = memory[:, :, :self.d_model]
    #
    #     return memory

    def forward_step(self, input, memory):
        # input: b * 768
        # memory: b * 256
        # memory = memory.reshape(-1, self.num_slots, self.d_model)
        input = self.proj1(input).unsqueeze(1)

        q = memory
        k = torch.cat([memory, input], 1)
        v = torch.cat([memory, input], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        # next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        outputs = []
        memory = memory.unsqueeze(1)
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.cat(outputs, dim=1)
        # memory = self.forward_step(inputs, memory)

        return outputs


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)).squeeze())


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.eps = eps
        self.mlp_gamma = nn.Sequential(nn.Linear(rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(d_model, d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)  # 2 * 1
        std = x.std(-1, keepdim=True)  # 2 * 1
        delta_gamma = self.mlp_gamma(memory)  # 2 * 768
        delta_beta = self.mlp_beta(memory)  # 2* 768
        gamma_hat = self.gamma.clone()  # 90
        beta_hat = self.beta.clone()  # 90
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)  # 2 * 90
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)  # 2 * 90 * 768
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0) # 2 * 90
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)  # 2 * 90 * 768
        gamma_hat += delta_gamma  # 2 * 90
        beta_hat += delta_beta  # 2 * 90
        # x: 2 * 90 * 768 memory: 2 * 256
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class DecoderLayer_memory(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_d_model):
        super(DecoderLayer_memory, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_d_model), 3)

    def forward(self, x, hidden_states, text_mask, image_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, text_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, image_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)


class Decoder_memory(nn.Module):
    def __init__(self, layer, N, rm):
        super(Decoder_memory, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.rm = rm

    def forward(self, x, hidden_states, memory, text_mask=None, image_mask=None):
        memory = self.rm(x, memory)
        for layer in self.layers:
            x = layer(x, hidden_states, text_mask, image_mask, memory)
        return self.norm(x)