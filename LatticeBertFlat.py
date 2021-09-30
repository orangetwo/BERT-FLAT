# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/9/27 3:58 下午
# @File    : LatticeBertFlat.py
import collections
import math
import torch
import copy

from torch import nn

from Modules import Transformer_Encoder, get_embedding
from utils import MyDropout, get_crf_zero_init

from fastNLP.core import seq_len_to_mask


class Lattice_Transformer_SeqLabel(nn.Module):
	def __init__(self,
	             lattice_embed,
	             bigram_embed,
	             hidden_size,
	             label_size,
	             num_heads,
	             num_layers,
	             learnable_position,
	             add_position,
	             layer_preprocess_sequence,
	             layer_postprocess_sequence,
	             ff_size=-1,
	             scaled=True,
	             dropout=None,
	             vocabs=None,
	             rel_pos_shared=True,
	             max_seq_len=-1,
	             k_proj=True,
	             q_proj=True,
	             v_proj=True,
	             r_proj=True,
	             self_supervised=False,
	             attn_ff=True,
	             pos_norm=False,
	             ff_activate='relu',
	             rel_pos_init=0,
	             embed_dropout_pos='0',
	             four_pos_shared=True,
	             four_pos_fusion=None,
	             four_pos_fusion_shared=True,
	             bert_embedding=None,
	             ):
		"""
		以下 transformer 简写为 tf
		这里说的 P为 paper中的 公式 9, 10
		:param lattice_embed: char + lexicon 对应的 embedding
		:param bigram_embed: bigram 对应的embedding 如果 没有提供可以稍微 None
		:param hidden_size: 传入 FLAT 的 tensor dimension
		:param label_size: 类别标签的个数
		:param num_heads: Transformer_Encoder 中 tf block 中 multi-heads self-attention 中heads 的设定
		:param num_layers: Transformer_Encoder 中 tf block 的个数
		:param learnable_position: 位置编码 是否可训练 默认 False
		:param add_position:
		:param layer_preprocess_sequence: 对输入的三种处理, 1. 残差 2.dropout 3.layer norm
		:param layer_postprocess_sequence: 对输入的三种处理, 1. 残差 2.dropout 3.layer norm
		:param ff_size: Transformer_Encoder 中 tf block 中 FFN 中的 维度, bert中 为 768 * 4
		:param scaled: 是否对 QK的值进行 放缩
		:param dropout: 是否 dropout
		:param vocabs: 词典, 词典中至少要提供 lattice
		:param rel_pos_shared: 是否对 四个P矩阵之间参数共享, 注意在P矩阵不可学习的情况下, 四个P矩阵参数共享无意义,因其初始化一样
		:param max_seq_len: 序列的最大长度
		:param k_proj: 是否对 tf 中的 key 进行映射 默认 True
		:param q_proj: 是否对 tf 中的 query 进行映射 默认 True
		:param v_proj: 是否对 tf 中的 value 进行映射 默认 True
		:param r_proj: 是否对 tf 中的 位置编码 进行映射 默认 True
		:param self_supervised:
		:param attn_ff: 是否对 multi-head self-attention 中 atten_score * value  进行映射
		:param pos_norm: 初始化后 是否对 P矩阵进行归一化
		:param ff_activate: tf中 激活函数
		:param rel_pos_init:    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
                                如果是1，那么就按-max_len,max_len来初始化
		:param embed_dropout_pos:  如果是0，就直接在embed后dropou t，是1就在embed变成hidden size之后再dropout，

		:param four_pos_shared: 四个P矩阵之间是否进行参数共享,默认情况下,由于参数不可学习,初始化方式又一样,效果可不是跟参数共享一样嘛
		:param four_pos_fusion: 四个P矩阵的融合方式, 对应 paper中公式 8
		:param four_pos_fusion_shared: 每层tf block之间的 融合 四个P矩阵 的参数(W(r)) 是否共享, 默认下tf只有1层,所以共不共享无所谓
										需要注意的 跨层之间的 四个P是参数共享的, 参考transformer xl
		:param bert_embedding: token 经过 bert 后的 embedding
		"""

		super().__init__()

		assert bert_embedding is not None, f" bert embedding 为空, 注意模型为 BERT-FLAT !!!"
		self.use_bigram = (bigram_embed is not None)

		self.bert_embedding = bert_embedding

		self.four_pos_fusion_shared = four_pos_fusion_shared
		self.four_pos_shared = four_pos_shared
		self.lattice_embed = lattice_embed
		self.bigram_embed = bigram_embed
		self.hidden_size = hidden_size
		self.label_size = label_size
		self.num_heads = num_heads
		self.num_layers = num_layers

		self.four_pos_fusion = four_pos_fusion
		self.learnable_position = learnable_position
		self.add_position = add_position

		self.self_supervised = self_supervised
		self.vocabs = vocabs
		self.attn_ff = attn_ff
		self.pos_norm = pos_norm
		self.ff_activate = ff_activate
		self.rel_pos_init = rel_pos_init
		self.embed_dropout_pos = embed_dropout_pos

		if max_seq_len < 0:
			ValueError(f'max_seq_len should be set ')

		self.max_seq_len = max_seq_len

		self.k_proj = k_proj
		self.q_proj = q_proj
		self.v_proj = v_proj
		self.r_proj = r_proj

		"""
		这里需要注意一件事, 对于绝对位置编码,bert和tf使用pe仅在第0 tf block层前使用
		而使用相对位置编码时,大部分模型在每层tf block都要输入位置信息
		"""

		"""
		创建 paper中的P矩阵, P矩阵有四个, 即paper中的d(hh), d(ht), d(th), d(ht)
		提供了2个选择, 1. 四个P是否参数共享, 2. 四个P是否为可训练参数(默认下四个P是不可训练的)
		"""

		pe = get_embedding(max_seq_len, hidden_size, rel_pos_init=self.rel_pos_init)
		pe_sum = pe.sum(dim=-1, keepdim=True)

		if self.pos_norm:
			with torch.no_grad():
				pe = pe / pe_sum
		self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)

		# 四个P矩阵参数共享
		if self.four_pos_shared:
			self.pe_ss = self.pe
			self.pe_se = self.pe
			self.pe_es = self.pe
			self.pe_ee = self.pe
		else:
			self.pe_ss = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
			self.pe_se = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
			self.pe_es = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
			self.pe_ee = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)

		self.layer_preprocess_sequence = layer_preprocess_sequence
		self.layer_postprocess_sequence = layer_postprocess_sequence

		if ff_size == -1:
			ff_size = self.hidden_size
		self.ff_size = ff_size
		# 是否对Q乘以K的值进行放缩 (attention score的计算)
		self.scaled = scaled

		if dropout is None:
			self.dropout = collections.defaultdict(int)
		else:
			self.dropout = dropout

		"""
		每个token最终的对应的embdding实际上可有三个embedding拼接(concat)
		finally embedding = bert embedding + lattice embedding(pretrain) + bigram embedding
		"""
		if self.use_bigram:
			self.bigram_size = self.bigram_embed.embedding.weight.size(1)
			self.char_input_size = self.lattice_embed.embedding.weight.size(1) \
			                       + self.bigram_embed.embedding.weight.size(1)
		else:
			self.char_input_size = self.lattice_embed.embedding.weight.size(1)

		# 加入 bert embedding size
		self.char_input_size += self.bert_embedding._embed_size

		self.lex_input_size = self.lattice_embed.embedding.weight.size(1)

		self.embed_dropout = MyDropout(self.dropout['embed'])
		self.gaz_dropout = MyDropout(self.dropout['gaz'])
		self.char_proj = nn.Linear(self.char_input_size, self.hidden_size)
		self.lex_proj = nn.Linear(self.lex_input_size, self.hidden_size)

		# encoder 的结构为 tf + crf
		# paper中的 R矩阵 都在 encoder 内部结构
		self.encoder = Transformer_Encoder(self.hidden_size,
		                                   self.num_heads,
		                                   self.num_layers,
		                                   learnable_position=self.learnable_position,
		                                   add_position=self.add_position,
		                                   layer_preprocess_sequence=self.layer_preprocess_sequence,
		                                   layer_postprocess_sequence=self.layer_postprocess_sequence,
		                                   dropout=self.dropout,
		                                   scaled=self.scaled,
		                                   ff_size=self.ff_size,
		                                   max_seq_len=self.max_seq_len,
		                                   pe=self.pe,
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
		                                   four_pos_fusion=self.four_pos_fusion,
		                                   four_pos_fusion_shared=self.four_pos_fusion_shared)

		self.output_dropout = MyDropout(self.dropout['output'])

		self.output = nn.Linear(self.hidden_size, self.label_size)
		if self.self_supervised:
			self.output_self_supervised = nn.Linear(self.hidden_size, len(vocabs['char']))
			print('self.output_self_supervised:{}'.format(self.output_self_supervised.weight.size()))

		self.crf = get_crf_zero_init(self.label_size)
		self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

	def forward(self, lattice, bigrams, seq_len, lex_num, pos_s, pos_e,
	            target, chars_target=None):  # 这里的名称必须和DataSet中相应的field对应
		"""
		bigrams 和 lattice 存在这样的关系:
		eg: bigrams 对应 [['我爱', '爱南', '南京', '京市', '市长', '长江', '江大', '大桥', '桥<end>'],
							['重庆', '庆人', '人和', '和药', '药店', '店<end>','<pad>'*3]]
			lattice 对应 [['我', '爱', '南', '京', '市', '长', '江', '大', '桥', '南京', '南京市', '市长', '长江大桥', '大桥'],
							['重', '庆', '人', '和', '药', '店', '重庆', '人和药店', '药店','<pad>'*5]]
			seq_len = [9, 6]
		可以看到 lattice 比 bigrams 长的部分是 对应的lexicons

		"""
		batch_size = lattice.size(0)
		max_seq_len_and_lex_num = lattice.size(1)
		max_seq_len = bigrams.size(1)

		# raw_embed 是字和词的pretrain的embedding，但是是分别trian的，所以需要区分对待
		raw_embed = self.lattice_embed(lattice)

		"""
		max_seq_len_and_lex_num 大于等于 max_seq_len
		故要补齐到 max_seq_len_and_lex_num, 这里补齐的方法很简单在第二个维度加0
		比如 (2,3,5) -> (2,5,5)
		"""
		if self.use_bigram:
			bigrams_embed = self.bigram_embed(bigrams)
			bigrams_embed = torch.cat([bigrams_embed,
			                           torch.zeros(size=[batch_size, max_seq_len_and_lex_num - max_seq_len,
			                                             self.bigram_size]).to(bigrams_embed)], dim=1)
			raw_embed_char = torch.cat([raw_embed, bigrams_embed], dim=-1)
		else:
			raw_embed_char = raw_embed

		"""
		char_for_bert 是不含 lexicon信息的
		eg: char_for_bert在上面的例子中 对应为 [['我', '爱', '南', '京', '市', '长', '江', '大', '桥']
												['重', '庆', '人', '和', '药', '店','<pad>'*3]]
		"""
		bert_pad_length = lattice.size(1) - max_seq_len
		char_for_bert = lattice[:, :max_seq_len]
		# 在上面的例子中 mask : torch.Size([2, 9])
		# tensor([[True, True, True, True, True, True, True, True, True],
		#         [True, True, True, True, True, True, False, False, False]])
		mask = seq_len_to_mask(seq_len).bool()
		char_for_bert = char_for_bert.masked_fill((~mask), self.vocabs['lattice'].padding_idx)
		bert_embed = self.bert_embedding(char_for_bert)
		bert_embed = torch.cat([bert_embed,
		                        torch.zeros(size=[batch_size, bert_pad_length, bert_embed.size(-1)],
		                                    device=bert_embed.device,
		                                    requires_grad=False)], dim=-2)
		raw_embed_char = torch.cat([raw_embed_char, bert_embed], dim=-1)

		# embedding dropout
		if self.embed_dropout_pos == '0':
			raw_embed_char = self.embed_dropout(raw_embed_char)
			raw_embed = self.gaz_dropout(raw_embed)

		# char_input_size -> self.hidden_size
		# embed_char (batch size, max_seq_len_and_lex_num, self.hidden_size)
		embed_char = self.char_proj(raw_embed_char)

		# 在上面的例子中 则为 torch.Size([2, 14])
		# tensor([[True, True, True, True, True, True, True, True, True, False,
		#          False, False, False, False],
		#         [True, True, True, True, True, True, False, False, False, False,
		#          False, False, False, False]])
		char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
		embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

		# lex_input_size -> hidden_size
		embed_lex = self.lex_proj(raw_embed)

		# 在上面的例子中 则为 torch.Size([2, 14])
		# tensor([[False, False, False, False, False, False, False, False, False, True,
		#          True, True, True, True],
		#         [False, False, False, False, False, False, True, True, True, False,
		#          False, False, False, False]])
		lex_mask = (seq_len_to_mask(seq_len + lex_num).bool() ^ char_mask.bool())
		embed_lex.masked_fill_(~lex_mask.unsqueeze(-1), 0)

		assert char_mask.size(1) == lex_mask.size(1), f" char_mask.size(1) don't match lex_mask.size(1)"

		embedding = embed_char + embed_lex

		encoded = self.encoder(embedding, seq_len, lex_num=lex_num, pos_s=pos_s, pos_e=pos_e)

		if hasattr(self, 'output_dropout'):
			encoded = self.output_dropout(encoded)

		encoded = encoded[:, :max_seq_len, :]
		pred = self.output(encoded)

		mask = seq_len_to_mask(seq_len).bool()

		if self.training:
			loss = self.crf(pred, target, mask).mean(dim=0)
			if self.self_supervised:

				chars_pred = self.output_self_supervised(encoded)
				chars_pred = chars_pred.view(size=[batch_size * max_seq_len, -1])
				chars_target = chars_target.view(size=[batch_size * max_seq_len])
				self_supervised_loss = self.loss_func(chars_pred, chars_target)

				loss += self_supervised_loss
			return {'loss': loss}
		else:
			pred, path = self.crf.viterbi_decode(pred, mask)
			result = {'pred': pred}
			if self.self_supervised:
				chars_pred = self.output_self_supervised(encoded)
				result['chars_pred'] = chars_pred

			return result
