# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/9/26 11:01 上午
# @File    : AddLattice.py
import torch
from fastNLP.embeddings import StaticEmbedding

from load_data import load_weibo_ner
from utils import Trie, extract_word_list
from functools import partial
from fastNLP.core import Vocabulary
from fastNLP import DataSet


def equip_chinese_ner_with_lexicon(datasets,
                                   vocabs,
                                   embeddings,
                                   w_list,
                                   word_embedding_path=None,
                                   only_lexicon_in_train=False,
                                   word_char_mix_embedding_path=None,  # 字和词的embedding信息
                                   lattice_min_freq=1,
                                   only_train_min_freq=0):
	"""
	datasets {dict: 3}: {'train':{'chars':, 'target':, 'bigrams':,'seq_len'} , 'test': , 'dev':}
	vocab {dict: 3}: {'char':由datasets中的 'chars'构建, 'label':,'bigram':由datasets中的'bigrams'构建}

	embeddings {dict 1} : {'char': embedding}
	"""

	if only_lexicon_in_train:
		print(f'已支持只加载在trian中出现过的词汇')

	def get_skip_path(chars, w_trie):
		sentence = ''.join(chars)
		result = w_trie.get_lexicon(sentence)
		# print(result)

		return result

	a = DataSet()
	w_trie = Trie()
	for w in w_list:
		w_trie.insert(w)

	if only_lexicon_in_train:
		lexicon_in_train = set()
		for s in datasets['train']['chars']:
			lexicon_in_s = w_trie.get_lexicon(s)
			for s, e, lexicon in lexicon_in_s:
				lexicon_in_train.add(''.join(lexicon))

		print('lexicon in train:{}'.format(len(lexicon_in_train)))
		print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
		w_trie = Trie()
		for w in lexicon_in_train:
			w_trie.insert(w)

	import copy

	# lexicons format ；[]
	for k, v in datasets.items():
		v.apply_field(partial(get_skip_path, w_trie=w_trie), 'chars', 'lexicons')
		v.apply_field(copy.copy, 'chars', 'raw_chars')
		v.add_seq_len('lexicons', 'lex_num')
		v.apply_field(lambda x: list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
		v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')

	def concat(ins):
		chars = ins['chars']
		lexicons = ins['lexicons']
		result = chars + list(map(lambda x: x[2], lexicons))
		"""
		result = ['我','爱','南','京','南京']
		"""
		return result

	def get_pos_s(ins):
		lex_s = ins['lex_s']
		seq_len = ins['seq_len']
		pos_s = list(range(seq_len)) + lex_s

		return pos_s

	def get_pos_e(ins):
		lex_e = ins['lex_e']
		seq_len = ins['seq_len']
		pos_e = list(range(seq_len)) + lex_e

		return pos_e

	# v.lattice 为 [['我','爱','南','京','南京'],[...]]
	for k, v in datasets.items():
		v.apply(concat, new_field_name='lattice')
		v.set_input('lattice')
		v.apply(get_pos_s, new_field_name='pos_s')
		v.apply(get_pos_e, new_field_name='pos_e')
		v.set_input('pos_s', 'pos_e')

	# vocabs {dict: 3} {'char':..., 'label':..., 'bigram':...}
	word_vocab = Vocabulary()
	word_vocab.add_word_lst(w_list)
	vocabs['word'] = word_vocab

	lattice_vocab = Vocabulary()
	lattice_vocab.from_dataset(datasets['train'], field_name='lattice',
	                           no_create_entry_dataset=[v for k, v in datasets.items() if k != 'train'])
	vocabs['lattice'] = lattice_vocab

	if word_embedding_path is not None:
		word_embedding = StaticEmbedding(word_vocab, word_embedding_path, word_dropout=0)
		embeddings['word'] = word_embedding

	if word_char_mix_embedding_path is not None:
		lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path, word_dropout=0.01,
		                                    min_freq=lattice_min_freq, only_train_min_freq=only_train_min_freq)
		embeddings['lattice'] = lattice_embedding
	else:
		embeddings['lattice'] = StaticEmbedding(lattice_vocab, embedding_dim=50)

	vocabs['char'].index_dataset(*(datasets.values()),
	                             field_name='chars', new_field_name='chars')
	vocabs['bigram'].index_dataset(*(datasets.values()),
	                               field_name='bigrams', new_field_name='bigrams')
	vocabs['label'].index_dataset(*(datasets.values()),
	                              field_name='target', new_field_name='target')
	vocabs['lattice'].index_dataset(*(datasets.values()),
	                                field_name='lattice', new_field_name='lattice')

	return datasets, vocabs, embeddings


if __name__ == '__main__':
	ds, vb, ed = load_weibo_ner('./data', unigram_embedding_path='./data/gigaword_chn.all.a2b.uni.ite50.vec',index_token=False)
	w_list = extract_word_list('/Users/orange/Desktop/Github/Flat-ner/V1/data/wordsListdemo.txt')
	datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(ds, vb, ed, w_list,word_embedding_path=None)
	from fastNLP.embeddings import BertEmbedding
	# model path : /Users/orange/.fastNLP/embedding/bert-chinese-wwm
	bert_embedding = BertEmbedding(vocabs['lattice'], model_dir_or_name='cn-wwm', requires_grad=False,
	                               word_dropout=0.01)

	inputs = [[vocabs['lattice'][word] for word in ['我','爱','南','京',"南京"]]]
	inputs = torch.tensor(inputs, dtype=torch.long)
	print(inputs)
	# inputs 输入 bert 时,会被添加 'CLS'和'SEP',即 ['[CLS]','我','爱','南','京',"南京","[SEP]"]
	# "南京" 在 BertEmbedding 内部被处理时会被映射为 "南" 和"##京" 而不是 "南" 和 "京"
	x = bert_embedding(inputs)
	# torch.Size([1, 5, 768])
	print(x.shape)

	"""
	逻辑:
		tokens 通过 vocab[token] 转化为 index, 在bert中这个index 存在另外一个 映射 index -> bert vocab
	"""