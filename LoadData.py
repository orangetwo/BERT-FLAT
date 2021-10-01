# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/9/25 2:59 下午
# @File    : LoadData.py
import os

from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from fastNLP.io.loader import ConllLoader


def load_weibo_ner(path, unigram_embedding_path=None, bigram_embedding_path=None, index_token=True,
                   char_min_freq=1, bigram_min_freq=1, only_train_min_freq=False, char_word_dropout=0.01):
	"""
    :param path: 数据所在的文件夹路径
    :param unigram_embedding_path: 存放 char embedding 文件的路径
    :param bigram_embedding_path:  存在 bigram embedding 文件的路径
    :param index_token: 是否要把 tokens 转化为 indices
    :param char_min_freq: 构建词典时, 最小词频
    :param bigram_min_freq: 构建词典时, 最小词频
    :param only_train_min_freq: 是否 只限制在train里面的词语使用min_freq筛选
    :param char_word_dropout: dropout
    :return: 三个字典, 第一个字典是数据集对应的token list, 第二个对应 char/bigram 的字典, 第三个 对应字典的 embedding
    """
	# 准备数据，诸如数据地址等
	# 关于ConllLoader详细结构 请查阅
	# https://github.com/fastnlp/fastNLP/blob/b127963f213226dc796720193965d86dface07d5/fastNLP/io/loader/conll.py#L28
	loader = ConllLoader(['chars', 'target'])
	# train_path = os.path.join(path, 'weiboNER_2nd_conll.train')
	# dev_path = os.path.join(path, 'weiboNER_2nd_conll.dev')
	# test_path = os.path.join(path, 'weiboNER_2nd_conll.test')

	# 用下面三个数据集进行测试
	train_path = os.path.join(path, 'train.demo')
	dev_path = os.path.join(path, 'dev.demo')
	test_path = os.path.join(path, 'test.demo')

	paths = {'train': train_path, 'dev': dev_path, 'test': test_path}

	# 构建datasets
	# 字典！！！ 但是需要注意的是：datasets 中的每一项都是一个(fastNLP)中 DataSet 类的实例
	datasets = {}
	# eg: k -> 'train', v -> './data/weiboNER_2nd_conll.train'
	for k, v in paths.items():
		# 这里每次创建一个DataSet()类
		bundle = loader.load(v)
		# 固定的 train 为参数，是因为bundle 这个实例的设置，它是把数据都放到 train 这个里面了
		datasets[k] = bundle.datasets['train']

	# datasets {dict: 3}  {'train': DataSet,'test': DataSet, 'dev': DataSet}
	trainData = datasets['train']

	"""
    datasets['train'] 中的数据长成下面这样，
        +-----------------------------------------------------------+-----------------------------------------------------------+
        | chars                                                     | target                                                    |
        +-----------------------------------------------------------+-----------------------------------------------------------+
        | ['科', '技', '全', '方', '位', '资', '讯', '智', '能',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['对', '，', '输', '给', '一', '个', '女', '人', '，',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'B-PER.NOM', 'I-PER.NOM... |
        | ['今', '天', '下', '午', '起', '来', '看', '到', '外',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['今', '年', '拜', '年', '不', '短', '信', '，', '就',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['浑', '身', '酸', '疼', '，', '两', '腿', '无', '力',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['明', '显', '紧', '张', '状', '态', '没', '出', '来',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['三', '十', '年', '前', '，', '老', '爹', '带', '我',...  | ['O', 'O', 'O', 'O', 'O', 'B-PER.NOM', 'I-PER.NOM', 'O... |
        | ['好', '活', '动', '呀', '，', '给', '力', '的', '商',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['人', '生', '如', '戏', '，', '导', '演', '是', '自',...  | ['O', 'O', 'O', 'O', 'O', 'B-PER.NOM', 'I-PER.NOM', 'O... |
        | ['听', '说', '小', '米', '开', '卖', '了', '，', '刚',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ...                                                       | ...                                                       |
        +-----------------------------------------------------------+-----------------------------------------------------------+

        这个是 复旦大学开源工具fastNLP 中DataSet 的类型，其详细文档可参考：
        https://fastnlp.readthedocs.io/zh/latest/tutorials/tutorial_1_data_preprocess.html

    """

	# 通过DataSet构建词典
	vocabs = {}
	# 通过以下链接可以查看fastnlp构建词典过程
	# https://fastnlp.readthedocs.io/zh/latest/tutorials/tutorial_2_vocabulary.html
	char_vocab = Vocabulary()
	bigram_vocab = Vocabulary()
	label_vocab = Vocabulary()

	for k, v in datasets.items():  # 处理键值对
		"""
        apply_field() 方法是fastNLP 中的一个处理DataSet 实例的方法
        对v.field_arrays['chars']做处理,并把处理后的值赋给v.field_arrays['chars']
        同理，第二个(get_bigrams,'chars','bigrams') 是根据 chars 这个列的值，新建bigrams这一列, 紧挨着的两个字拼接
        
        通过 v.field_arrays['chars'] 和 v.field_arrays['bigrams'] 可以构建 1-gram 字典和 2-gram 字典
        """
		v.apply_field(lambda x: [w[0] for w in x], 'chars', 'chars')
		v.apply_field(get_bigrams, 'chars', 'bigrams')

	# 构建词典
	# no_create_entry_dataset 中的entity也会被放进词典中
	# 但在后续利用static embedding为char_vocab创建embedding 并不会为该entity创建对应的vector
	# 逻辑上看不太懂 为啥还能 len(vocab) > embedding.size(0)
	# 在这种逻辑下面 no_create_entry_dataset 只是提供在datasets['train'] entity的词频的更新
	char_vocab.from_dataset(datasets['train'], field_name='chars',
	                        no_create_entry_dataset=[datasets['dev'], datasets['test']])
	label_vocab.from_dataset(datasets['train'], field_name='target')

	for k, v in datasets.items():
		# v.set_pad_val('target',-100)
		v.add_seq_len('chars', new_field_name='seq_len')

	bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
	                          no_create_entry_dataset=[datasets['dev'], datasets['test']])

	# 将 str 转化为 index
	if index_token:
		char_vocab.index_dataset(*list(datasets.values()), field_name='chars', new_field_name='chars')
		bigram_vocab.index_dataset(*list(datasets.values()), field_name='bigrams', new_field_name='bigrams')
		label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

	# vocabs 的构造和 datasets 的构造原理都是相同的
	# 二者都是字典，不同的键值对应着不同的数据信息
	vocabs['char'] = char_vocab
	vocabs['label'] = label_vocab
	vocabs['bigram'] = bigram_vocab

	# 构建embedding
	embeddings = {}
	"""
	if unigram_embedding_path is not None:

		unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
		                                    word_dropout=char_word_dropout,
		                                    min_freq=char_min_freq, only_train_min_freq=only_train_min_freq,
		                                    only_use_pretrain_word=False)
		embeddings['char'] = unigram_embedding
	"""
	"""
	以下情况仅限于 StaticEmbedding
	1. 如果指定了 model_dir_or_name , 对于 model_dir_or_name 中每个token 如果其出现在 char_vocab
		保留这个token 及其对应 vector, 未出现的丢弃
	2. 在 only_use_pretrain_word 指定为False(默认)的情况下, 对于 未出现在 model_dir_or_name 且出现在
		datasets['train']['chars']中的token保留,并初始化对应的vector , 将1 中产生的vector和此步骤产生的vector
		合并为 embedding
	注意最终在上面的设定中, 最终embedding对应的 token 可能存在下列情况
		1. 在 model_dir_or_name中, 不在datasets['train']['chars'], 在 datasets['test/dev']['chars']
		2. 不在model_dir_or_name中, 在datasets['train']['chars'], 不在 datasets['test/dev']['chars']
		3. etc
	不会存在下列情况:
		1. 不在model_dir_or_name中, 不在datasets['train']['chars'], 在 datasets['test/dev']['chars']
	"""

	if bigram_embedding_path is not None:
		bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
		                                   word_dropout=0.01, min_freq=bigram_min_freq,
		                                   only_train_min_freq=only_train_min_freq)
	else:
		bigram_embedding = StaticEmbedding(bigram_vocab, embedding_dim=50)
	embeddings['bigram'] = bigram_embedding

	return datasets, vocabs, embeddings


def get_bigrams(words):
	result = []
	for i, w in enumerate(words):
		if i != len(words) - 1:
			result.append(words[i] + words[i + 1])
		else:
			result.append(words[i] + '<end>')

	return result


if __name__ == '__main__':
	load_weibo_ner('./data', unigram_embedding_path='./data/gigaword_chn.all.a2b.uni.ite50.vec', index_token=False)
