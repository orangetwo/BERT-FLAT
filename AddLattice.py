# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/9/26 11:01 上午
# @File    : AddLattice.py
import torch
from fastNLP.embeddings import StaticEmbedding

from LoadData import load_weibo_ner
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
	# 这里 默认的 数据集路径 data/train.demo data/test.demo data/dev.demo
	ds, vb, ed = load_weibo_ner('./data', unigram_embedding_path='./data/gigaword_chn.all.a2b.uni.ite50.vec',
	                            index_token=False)
	w_list = extract_word_list('/Users/orange/Desktop/Github/Flat-ner/V1/data/wordsListdemo.txt')
	datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(ds, vb, ed, w_list, word_embedding_path=None)
	from fastNLP.embeddings import BertEmbedding

	# model path : /Users/orange/.fastNLP/embedding/bert-chinese-wwm
	bert_embedding = BertEmbedding(vocabs['lattice'], model_dir_or_name='cn-wwm', requires_grad=False,
	                               word_dropout=0.01)

	inputs = [[vocabs['lattice'][word] for word in ['我', '爱', '南', '京', "南京"]]]
	inputs = torch.tensor(inputs, dtype=torch.long)
	print(inputs)
	# inputs 输入 bert 时,会被添加 'CLS'和'SEP',即 ['[CLS]','我','爱','南','京',"南京","[SEP]"]
	# "南京" 在 BertEmbedding 内部被处理时会被映射为 "南" 和"##京" 而不是 "南" 和 "京"
	x = bert_embedding(inputs)
	# torch.Size([1, 5, 768])
	print(x)

	"""
	bert_embedding = BertEmbedding(vocabs['lattice'], model_dir_or_name='cn-wwm', requires_grad=False,
	                               word_dropout=0.01)
	inputs = [[vocabs['lattice'][word] for word in ['我','爱','南','京',"南京"]]]
	inputs = torch.tensor(inputs, dtype=torch.long)
	x = bert_embedding(inputs)
	
	BertEmbedding会根据vocabs['lattice']构建词典, 注意 vocabs['lattice'].word2idx
	BertEmbedding会仿照vocabs['lattice'].idx2word 构建自己的映射 BertEmbedding._BertWordModel.word_to_wordpieces
	
	vocabs['lattice'].word2idx : 
	{'<pad>': 0,
	 '<unk>': 1,
	 '我': 2,
	 '爱': 3,
	 '市': 4,
	 '长': 5,
	 '江': 6,
	 '大': 7,
	 '桥': 8,
	 '市长': 9,
	 '长江大桥': 10,
	 '大桥': 11,
	 '重': 12,
	 '庆': 13,
	 '人': 14,
	 '和': 15,
	 '药': 16,
	 '店': 17,
	 '重庆': 18,
	 '人和药店': 19,
	 '药店': 20,
	 '南': 21,
	 '京': 22,
	 '南京': 23,
	 '南京市': 24,
	 '阳': 25,
	 '南阳': 26,
	 '南阳市': 27,
	 '武': 28,
	 '汉': 29,
	 '武汉': 30}

	BertEmbedding._BertWordModel.word_to_wordpieces : 
	array([list([0]), list([100]), list([2769]), list([4263]), list([2356]),
       list([7270]), list([3736]), list([1920]), list([3441]),
       list([2356, 20327]), list([7270, 16793, 14977, 16498]),
       list([1920, 16498]), list([7028]), list([2412]), list([782]),
       list([1469]), list([5790]), list([2421]), list([7028, 15469]),
       list([782, 14526, 18847, 15478]), list([5790, 15478]),
       list([1298]), list([776]), list([1298, 13833]),
       list([1298, 13833, 15413]), list([7345]), list([1298, 20402]),
       list([1298, 20402, 15413]), list([3636]), list([3727]),
       list([3636, 16784])], dtype=object)
       
    故 对于 一个 输入 token 如"南"处理顺序为 先在 vocabs['lattice'].word2idx 找到其index 为 21 
    而 BertEmbedding._BertWordModel.word_to_wordpieces中第21个元素为 list([1298])
    1298 即为 "南"在bert词表中的位置
	
	inputs 输入 bert_embedding 时,会被添加 '[CLS]'和'[SEP]', 即 ['index([CLS])','index(我)','index(爱)','index(南)','index(京)',"index(南京)","index([SEP])"]
	
	得到input的表征：([1, 8, 768])
	tensor([[[ 0.0839,  0.2205,  0.2262,  ...,  0.8225,  0.2395, -0.2320],
         [ 1.0193,  0.4427, -0.5075,  ..., -0.2693,  0.3188, -0.4572],
         [ 1.1579, -0.1583, -0.6160,  ...,  0.0205,  0.4773, -0.4224],
         ...,
         [-0.6216, -0.0428,  0.7047,  ..., -0.6202,  0.6006, -0.6023],
         [-0.3761,  0.0325, -0.3896,  ...,  0.9522,  0.0729, -0.2888],
         [ 0.0975,  0.4757,  0.6848,  ..., -0.1029,  0.1026,  0.1724]]])
	
	去掉"[CLS]"和"[SEP]"的表征 : torch.Size([1, 6, 768])
	tensor([[[ 1.0193,  0.4427, -0.5075,  ..., -0.2693,  0.3188, -0.4572],
         [ 1.1579, -0.1583, -0.6160,  ...,  0.0205,  0.4773, -0.4224],
         [ 0.5190,  0.0286,  0.1981,  ..., -0.4006,  0.6613, -0.3119],
         [ 0.3893,  0.6789,  0.4636,  ...,  0.1320,  0.5017, -0.2525],
         [-0.6216, -0.0428,  0.7047,  ..., -0.6202,  0.6006, -0.6023],
         [-0.3761,  0.0325, -0.3896,  ...,  0.9522,  0.0729, -0.2888]]])
         
    上面 最后两行分别对应 "南"和"##京", 所以要恢复到原来的size 即 [1, 5, 768]
    这里恢复的策略有好几种 , bert_embedding 内部提供了几种 "first"(默认), "last", "max"等。
    这里以 "first"为例
    
    x：torch.Size([1, 1, 5, 768])
	
	tensor([[[[ 1.0193,  0.4427, -0.5075,  ..., -0.2693,  0.3188, -0.4572],
          [ 1.1579, -0.1583, -0.6160,  ...,  0.0205,  0.4773, -0.4224],
          [ 0.5190,  0.0286,  0.1981,  ..., -0.4006,  0.6613, -0.3119],
          [ 0.3893,  0.6789,  0.4636,  ...,  0.1320,  0.5017, -0.2525],
          [-0.6216, -0.0428,  0.7047,  ..., -0.6202,  0.6006, -0.6023]]]])
    
    可以看到所谓的 "first"其实就是只保留每个word切割为subword下第一个token的对应的 embedding
    放到这个例子就是 "南"和"##京"作为一个word切割后我们只保留第一个token 也就是 "南"对应的embedding
    
    
    在我们这个例子中 
    
    ['我','爱','南','京',"南京"] 在 vocabs['lattice']内 部索引表示为tensor([[ 2,  3, 21, 22, 23]])
    ['我','爱','南','京',"南京"] 在 bert_embedding 内部索引表示(进入bertModel前,加入了"[CLS]"和"[SEP]") tensor([[  101,  2769,  4263,  1298,   776,  1298, 13833,   102]])
	"""
