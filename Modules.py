# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/9/27 7:38 下午
# @File    : Modules.py

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import collections

from fastNLP.core.utils import seq_len_to_mask

from utils import MyDropout


def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """
    对应paper里的 式8, 式9, 式10 中的 P矩阵
    """
    num_embeddings = 2 * max_seq_len + 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)

    """
    paper中 公式 9 中d的取值策略
    如果是0，那么从 -max_len 到 max_len 的相对位置编码矩阵就按 0 - 2 * max_len 来初始化，
    如果是1，那么就按 -max_len, max_len 来初始化
    """
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len, max_seq_len + 1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


class Four_Pos_Fusion_Embedding(nn.Module):
    """
    此类主要用来提供四个位置编码的融合方式
    创建实例的时 需要传入 这四个矩阵 同时指定 融合方式
    使用时 需要传入 token 的 head[i], tail[i]
    eg : 如 paper 中的 Figure 2, 假设 batch size = 1,
            则有 pos_s = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2, 4]])
                pos_e = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 5, 5]])
    返回值 则对应paper中的 公式 8  的 R(i,j)
    """

    def __init__(self, four_pos_fusion, pe_ss, pe_se, pe_es, pe_ee, max_seq_len, hidden_size,**kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        # self.pe = pe
        self.four_pos_fusion = four_pos_fusion

        # 以下 主要对应 paper中的公式 8, 8中简单的拼接, 这里提供了五种融合的方式
        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size),
                                                    nn.ReLU(inplace=True))
        if self.four_pos_fusion == 'ff_linear':
            self.pos_fusion_forward = nn.Linear(self.hidden_size * 4, self.hidden_size)

        elif self.four_pos_fusion == 'ff_two':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size * 4, 4),
                                                nn.Softmax(dim=-1))
        elif self.four_pos_fusion == 'gate':
            self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size * 2, 4 * self.hidden_size))

    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)

        """
        pos_s 对应paper中 Figure2中的head
        pos_e 对应paper中 Figure2中的tail
        皆为 (batch size, sequence length)
        """
        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

        """
        pos_ss 对应paper中的公式4 
        (batch size, sequence length, sequence length)
        pos_ss 中 第i行第j列代表 head[i]-head[j]
        """

        # B prepare relative position encoding
        max_seq_len = pos_s.size(1)
        # rel_distance = self.seq_len_to_rel_distance(max_seq_len)

        # rel_distance_flat = rel_distance.view(-1)
        # rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        # rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])

        # (pos_ss).view(-1)  : (batch size * sequence length * sequence length)
        # pe_ss : (batch size, sequence length, sequence length, hidden dimension)
        pe_ss = self.pe_ss[pos_ss.view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_se = self.pe_se[pos_se.view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe_es[pos_es.view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[pos_ee.view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        # print('pe_ss:{}'.format(pe_ss.size()))

        # 下面对应 paper公式8
        if self.four_pos_fusion == 'ff':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_linear':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_two':
            pe_2 = torch.cat([pe_ss, pe_ee], dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_2)
        elif self.four_pos_fusion == 'attn':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            attn_score = self.pos_attn_score(pe_4)
            pe_4_unflat = self.w_r(pe_4.view(batch, max_seq_len, max_seq_len, 4, self.hidden_size))
            pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion
            if self.mode['debug']:
                print('pe_4照理说应该是 Batch * SeqLen * SeqLen * HiddenSize')
                print(pe_4_fusion.size())

        elif self.four_pos_fusion == 'gate':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            gate_score = self.pos_gate_score(pe_4).view(batch, max_seq_len, max_seq_len, 4, self.hidden_size)
            gate_score = F.softmax(gate_score, dim=-2)
            pe_4_unflat = self.w_r(pe_4.view(batch, max_seq_len, max_seq_len, 4, self.hidden_size))
            pe_4_fusion = (gate_score * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion

        return rel_pos_embedding


class MultiHead_Attention_Lattice_rel_save_gpumm(nn.Module):
    """
    对应 transformer xl 中的位置编码, 以及 multi-heads self-attention 的计算
    """
    def __init__(self, hidden_size, num_heads,
                 scaled=True, max_seq_len=-1,
                 k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 attn_dropout=None,
                 ff_final=True,
                 four_pos_fusion=None, *kwargs):
        """
        :param hidden_size: 输入的hidden state dimension , 比如 在bert中为 768
        :param num_heads:  多头的个数, 在 bert-base中 好像是 12
        :param pe:
        :param pe_ss:
        :param pe_se:
        :param pe_es:
        :param pe_ee:
        :param scaled:
        :param max_seq_len:
        :param dvc:
        :param mode:
        :param k_proj:
        :param q_proj:
        :param v_proj:
        :param r_proj:
        :param attn_dropout:
        :param ff_final:
        :param four_pos_fusion:
        """

        super().__init__()
        assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        # self.pe_ss = pe_ss
        # self.pe_se = pe_se
        # self.pe_es = pe_es
        # self.pe_ee = pe_ee

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len

        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size * 4, 4),
                                                nn.Softmax(dim=-1))

        elif self.four_pos_fusion == 'gate':
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size * 2, 4 * self.hidden_size))

        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))

        # self.pe = pe

        self.dropout = MyDropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, key, query, value, seq_len, lex_num, rel_pos_embedding, **kwargs):
        batch = key.size(0)

        if self.k_proj:
            key = self.w_k(key)
        if self.q_proj:
            query = self.w_q(query)
        if self.v_proj:
            value = self.w_v(value)
        if self.r_proj:
            rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)

        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [batch, max_seq_len, max_seq_len, self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)
        # #A
        # A_ = torch.matmul(query,key)
        # #C
        # # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        # key_for_c = key
        # C_ = torch.matmul(u_for_c, key)
        query_and_u_for_c = query + u_for_c

        # 对应 transformer xl 中的 A + C
        A_C = torch.matmul(query_and_u_for_c, key)

        # B
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # after above, query_for_b: batch * num_head * query_len * 1 * per_head_size
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        # B_ = torch.matmul(query_for_b,rel_pos_embedding_for_b).squeeze(-2)

        # D
        # rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: batch * query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        # v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        # D_ = torch.matmul(rel_pos_embedding_for_d,v_for_d).squeeze(-1).squeeze(-1).permute(0,3,1,2)

        query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_heads, 1, 1, self.per_head_size)

        # 对应 transformer xl 中的 B + D
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)
        # att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape

        attn_score_raw = A_C + B_D

        if self.scaled:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)

        # mask (batch size, 1 , 1 sequence length + lexicon number)
        mask = seq_len_to_mask(seq_len + lex_num).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)

        attn_score = F.softmax(attn_score_raw_masked, dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)

        if hasattr(self, 'ff_final'):
            print('ff_final!!')
            result = self.ff_final(result)

        return result

    def seq_len_to_rel_distance(self, max_seq_len):
        '''
        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index


class PositionWise_FeedForward(nn.Module):
    def __init__(self, sizes, dropout=None, ff_activate='relu'):
        super().__init__()
        self.num_layers = len(sizes) - 1
        for i in range(self.num_layers):
            setattr(self, 'w' + str(i), nn.Linear(sizes[i], sizes[i + 1]))

        if dropout is None:
            dropout = collections.defaultdict(int)

        self.dropout = MyDropout(dropout['ff'])
        self.dropout_2 = MyDropout(dropout['ff_2'])
        if ff_activate == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif ff_activate == 'leaky':
            self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, inp):
        output = inp
        for i in range(self.num_layers):
            if i != 0:
                output = self.activate(output)
            w = getattr(self, 'w' + str(i))
            output = w(output)
            if i == 0:
                output = self.dropout(output)
            if i == 1:
                output = self.dropout_2(output)

        return output


class Transformer_Encoder_Layer(nn.Module):

    def __init__(self, hidden_size, num_heads,
                 learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 max_seq_len=-1,
                 pe_ss=None, pe_se=None, pe_es=None, pe_ee=None,
                 k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 attn_ff=True, ff_activate='relu',
                 four_pos_shared=True, four_pos_fusion=None, four_pos_fusion_embedding=None
                 ):
        super().__init__()
        self.four_pos_fusion_embedding = four_pos_fusion_embedding
        self.four_pos_shared = four_pos_shared
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate

        if max_seq_len < 0:
            ValueError(f'max_seq_len should be set ')

        self.max_seq_len = max_seq_len

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        if self.four_pos_fusion_embedding is None:
            self.four_pos_fusion_embedding = \
                Four_Pos_Fusion_Embedding(self.four_pos_fusion, self.pe_ss, self.pe_se, self.pe_es, self.pe_ee,
                                          self.max_seq_len,)

        if dropout is None:
            dropout = collections.defaultdict(int)
        self.dropout = dropout

        if ff_size == -1:
            ff_size = hidden_size
        self.ff_size = ff_size
        # print('dropout:{}'.format(self.dropout))
        self.layer_preprocess = Layer_Process(self.layer_preprocess_sequence, self.hidden_size, self.dropout['pre'])
        self.layer_postprocess = Layer_Process(self.layer_postprocess_sequence, self.hidden_size, self.dropout['post'])

        self.attn = MultiHead_Attention_Lattice_rel_save_gpumm(self.hidden_size, self.num_heads,
                                                               scaled=self.scaled,
                                                               max_seq_len=self.max_seq_len,
                                                               k_proj=self.k_proj,
                                                               q_proj=self.q_proj,
                                                               v_proj=self.v_proj,
                                                               r_proj=self.r_proj,
                                                               attn_dropout=self.dropout['attn'],
                                                               ff_final=self.attn_ff,
                                                               four_pos_fusion=self.four_pos_fusion)

        self.ff = PositionWise_FeedForward([hidden_size, ff_size, hidden_size], self.dropout,
                                           ff_activate=self.ff_activate)

    def forward(self, inp, seq_len, lex_num=0, pos_s=None, pos_e=None, rel_pos_embedding=None):
        output = inp
        output = self.layer_preprocess(output)

        if rel_pos_embedding is None:
            """举一个例子 ,如 paper 中的 Figure 2, 假设 batch size = 1
                    则有
                    pos_s = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2, 4]])
                    pos_e = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 5, 5]])"""
            rel_pos_embedding = self.four_pos_fusion_embedding(pos_s, pos_e)
        # multi-head self attention
        output = self.attn(output, output, output, seq_len, pos_s=pos_s, pos_e=pos_e, lex_num=lex_num,
                           rel_pos_embedding=rel_pos_embedding)

        output = self.layer_postprocess(output)
        output = self.layer_preprocess(output)
        output = self.ff(output)
        output = self.layer_postprocess(output)

        return output


class Layer_Process(nn.Module):
    def __init__(self, process_sequence, hidden_size, dropout=0, ):
        super().__init__()
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        if 'd' in self.process_sequence:
            self.dropout = MyDropout(dropout)
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inp):
        """
        对输入的三种处理, 1. 参差 2.dropout 3.layer norm
        """
        output = inp
        for op in self.process_sequence:
            if op == 'a':
                output = output + inp
            elif op == 'd':
                output = self.dropout(output)
            elif op == 'n':
                output = self.layer_norm(output)

        return output


class Transformer_Encoder(nn.Module):
    def __init__(self, hidden_size,
                 num_heads,
                 num_layers,
                 learnable_position,
                 add_position,
                 layer_preprocess_sequence,
                 layer_postprocess_sequence,
                 dropout=None,
                 scaled=True,
                 ff_size=-1,
                 max_seq_len=-1,
                 pe_ss=None,
                 pe_se=None,
                 pe_es=None,
                 pe_ee=None,
                 k_proj=True,
                 q_proj=True,
                 v_proj=True,
                 r_proj=True,
                 attn_ff=True,
                 ff_activate='relu',
                 four_pos_shared=True,
                 four_pos_fusion=None,
                 four_pos_fusion_shared=True, **kwargs):
        """
       :param hidden_size: 输入 tf 的 hidden dimension
       :param num_heads:  tf 中 multi-heads
       :param num_layers:  几层 tf block
       :param learnable_position: P矩阵是否可以学习
       :param add_position:
       :param layer_preprocess_sequence: 对输入的三种处理, 1. 残差 2.dropout 3.layer norm
       :param layer_postprocess_sequence: 对输入的三种处理, 1. 残差 2.dropout 3.layer norm
       :param dropout: 是否 dropout
       :param scaled: 是否对 QK的值进行 放缩
       :param ff_size: Transformer_Encoder 中 tf block 中 FFN 中的 维度, bert中 为 768 * 4
       :param max_seq_len: 序列的最大长度
       :param pe_ss: 对应 paper 中的 Pd(hh)
       :param pe_se: 参考上面
       :param pe_es: 参考上面
       :param pe_ee: 参考上面
       :param k_proj: 是否对 tf 中的 key 进行映射 默认 True
       :param q_proj: 是否对 tf 中的 query 进行映射 默认 True
       :param v_proj: 是否对 tf 中的 value 进行映射 默认 True
       :param r_proj: 是否对 tf 中的 位置编码 进行映射 默认 True
       :param attn_ff: 是否对 multi-head self-attention 中 atten_score * value  进行映射
       :param ff_activate: tf中 激活函数
       :param four_pos_shared: 四个P之间共享
       :param four_pos_fusion: 指定融合方式
       :param four_pos_fusion_shared: 每层tf block之间的 融合 四个P矩阵 的参数(W(r)) 是否共享, 默认下tf只有1层,所以共不共享无所谓
										需要注意的 跨层之间的 四个P是参数共享的, 参考transformer xl
       :param kwargs:
        """

        super().__init__()
        self.four_pos_fusion_shared = four_pos_fusion_shared
        self.four_pos_shared = four_pos_shared
        self.four_pos_fusion = four_pos_fusion
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        """
        是否对位置编码矩阵进行参数共享:
            共享: 对每层transformer block 中的 位置编码 部分 进行参数共享
                由于参数共享 每层的 R(i,j) （paper中 公式8）都是相等的
            不共享: 要对每层的block 创建 R(i,j) 由于 每层的四个P矩阵初始化方式相同(默认下,这4个矩阵且不可被训练)
                融合方式也相同, 所以这里 造成的 R(i,j)不同的原因是 对应的 W(r)不同            
        """
        if self.four_pos_fusion_shared:
            self.four_pos_fusion_embedding = \
                Four_Pos_Fusion_Embedding(self.four_pos_fusion, self.pe_ss, self.pe_se, self.pe_es, self.pe_ee,
                                          self.max_seq_len, self.hidden_size, )

        self.num_heads = num_heads
        self.num_layers = num_layers

        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate

        if max_seq_len < 0:
            ValueError(f'max_seq_len should be set ')

        if dropout is None:
            dropout = collections.defaultdict(int)
        self.dropout = dropout

        if ff_size == -1:
            ff_size = hidden_size
        self.ff_size = ff_size

        for i in range(self.num_layers):
            setattr(self, 'layer_{}'.format(i), Transformer_Encoder_Layer(hidden_size, num_heads,
                                                                          learnable_position,
                                                                          add_position,
                                                                          layer_preprocess_sequence,
                                                                          layer_postprocess_sequence,
                                                                          dropout, scaled, ff_size,
                                                                          max_seq_len=self.max_seq_len,
                                                                          pe_ss=self.pe_ss,
                                                                          pe_se=self.pe_se,
                                                                          pe_es=self.pe_es,
                                                                          pe_ee=self.pe_ee,
                                                                          k_proj=self.k_proj,
                                                                          q_proj=self.q_proj,
                                                                          v_proj=self.v_proj,
                                                                          r_proj=self.r_proj,
                                                                          attn_ff=self.attn_ff,
                                                                          ff_activate=self.ff_activate,
                                                                          four_pos_shared=self.four_pos_shared,
                                                                          four_pos_fusion=self.four_pos_fusion,
                                                                          four_pos_fusion_embedding=self.four_pos_fusion_embedding
                                                                          ))

        self.layer_preprocess = Layer_Process(self.layer_preprocess_sequence, self.hidden_size)

    def forward(self, inp, seq_len, lex_num=0, pos_s=None, pos_e=None):
        output = inp

        """
        相对位置编码
        是否对 R(i,j)进行参数共享
            共享: 所有层 都用同一个 rel_pos_embedding
            不共享: 每个层自己用的rel_pos_embedding, 对应下面的none, 会在下面的now_layer自己创建 rel_pos_embedding
        """
        if self.four_pos_fusion_shared:
            rel_pos_embedding = self.four_pos_fusion_embedding(pos_s, pos_e)
        else:
            rel_pos_embedding = None

        for i in range(self.num_layers):
            now_layer = getattr(self, 'layer_{}'.format(i))
            output = now_layer(output, seq_len, lex_num=lex_num, pos_s=pos_s, pos_e=pos_e,
                               rel_pos_embedding=rel_pos_embedding)

        output = self.layer_preprocess(output)

        return output
