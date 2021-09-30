import argparse
import collections

import torch
from fastNLP import Trainer, LossInForward, SpanFPreRecMetric, AccuracyMetric
from torch import optim

from AddLattice import equip_chinese_ner_with_lexicon
from LatticeBertFlat import Lattice_Transformer_SeqLabel
from load_data import load_weibo_ner
from utils import extract_word_list
from fastNLP.embeddings import BertEmbedding

parser = argparse.ArgumentParser()
# -----------------------------------------如下一大部分全都是修改参数相关------------------------------------
# -----------------------------------------performance irrelevant-----------------------------------------
parser.add_argument('--update_every',type=int,default=1)
parser.add_argument('--status',choices=['train','test'],default='train')
parser.add_argument('--use_bert',type=int,default=1)
parser.add_argument('--only_bert',type=int,default=0)
parser.add_argument('--fix_bert_epoch',type=int,default=20)
parser.add_argument('--after_bert',default='mlp',choices=['lstm','mlp'])
parser.add_argument('--msg',default='11266')
parser.add_argument('--train_clip',default=False,help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0,type=int)
parser.add_argument('--gpumm',default=False,help='查看显存')
parser.add_argument('--see_convergence',default=False)
parser.add_argument('--see_param',default=False)
parser.add_argument('--test_batch', default=1)
parser.add_argument('--seed', default=1080956,type=int)
parser.add_argument('--test_train',default=False)
parser.add_argument('--number_normalized',type=int,default=0,
                    choices=[0,1,2,3],help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name',default='yj',choices=['lk','yj'])
parser.add_argument('--use_pytorch_dropout',type=int,default=0)
parser.add_argument('--char_min_freq',default=1,type=int)
parser.add_argument('--bigram_min_freq',default=1,type=int)
parser.add_argument('--lattice_min_freq',default=1,type=int)
parser.add_argument('--only_train_min_freq',default=True)
parser.add_argument('--only_lexicon_in_train',default=False)
parser.add_argument('--word_min_freq',default=1,type=int)


# hyper of training
parser.add_argument('--early_stop',default=25,type=int)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--batch', default=2, type=int) # 我修改了默认batch的大小
parser.add_argument('--optim', default='sgd', help='sgd|adam')
#parser.add_argument('--lr', default=6e-4, type=float)
parser.add_argument('--lr', default=6e-3, type=float)
parser.add_argument('--bert_lr_rate',default=0.05,type=float)
parser.add_argument('--embed_lr_rate',default=1,type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init',default='uniform',help='norm|uniform')
parser.add_argument('--self_supervised',default=False)
parser.add_argument('--weight_decay',default=0,type=float)
parser.add_argument('--norm_embed',default=True)
parser.add_argument('--norm_lattice_embed',default=True)
parser.add_argument('--warmup',default=0.1,type=float)

# hyper of model
parser.add_argument('--model',default='transformer',help='lstm|transformer')
parser.add_argument('--lattice',default=1,type=int)
parser.add_argument('--use_bigram', default=1,type=int)
parser.add_argument('--hidden', default=-1,type=int)
parser.add_argument('--ff', default=3,type=int)
parser.add_argument('--layer', default=1,type=int)
parser.add_argument('--head', default=8,type=int)
parser.add_argument('--head_dim',default=20,type=int)
parser.add_argument('--scaled',default=False)
parser.add_argument('--ff_activate',default='relu',help='leaky|relu')

parser.add_argument('--k_proj',default=False)
parser.add_argument('--q_proj',default=True)
parser.add_argument('--v_proj',default=True)
parser.add_argument('--r_proj',default=True)

parser.add_argument('--attn_ff',default=False)

# parser.add_argument('--rel_pos', default=False)
parser.add_argument('--use_abs_pos',default=False)
parser.add_argument('--use_rel_pos',default=True)
# 相对位置和绝对位置不是对立的，可以同时使用
parser.add_argument('--rel_pos_shared',default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm',default=False)
parser.add_argument('--rel_pos_init',default=1)
parser.add_argument('--four_pos_shared',default=True,help='只针对相对位置编码，指4个位置编码是不是共享权重')
parser.add_argument('--four_pos_fusion',default='ff_two',choices=['ff','attn','gate','ff_two','ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared',default=True,help='是不是要共享4个位置融合之后形成的pos')

parser.add_argument('--pre', default='')
parser.add_argument('--post', default='an')

over_all_dropout = -1
parser.add_argument('--embed_dropout_before_pos',default=False)
parser.add_argument('--embed_dropout', default=0.5,type=float)
parser.add_argument('--gaz_dropout',default=0.5,type=float)
parser.add_argument('--output_dropout', default=0.3,type=float)
parser.add_argument('--pre_dropout', default=0.5,type=float)
parser.add_argument('--post_dropout', default=0.3,type=float)
parser.add_argument('--ff_dropout', default=0.15,type=float)
parser.add_argument('--ff_dropout_2', default=-1,type=float,help='FF第二层过完后的dropout，之前没管这个的时候是0')
parser.add_argument('--attn_dropout',default=0,type=float)
parser.add_argument('--embed_dropout_pos',default='0')
parser.add_argument('--abs_pos_fusion_func',default='nonlinear_add',
                    choices=['add','concat','nonlinear_concat','nonlinear_add','concat_nonlinear','add_nonlinear'])


args = parser.parse_args()

if args.ff_dropout_2 < 0:
    args.ff_dropout_2 = args.ff_dropout

if over_all_dropout>0:
    args.embed_dropout = over_all_dropout
    args.output_dropout = over_all_dropout
    args.pre_dropout = over_all_dropout
    args.post_dropout = over_all_dropout
    args.ff_dropout = over_all_dropout
    args.attn_dropout = over_all_dropout


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dropout = collections.defaultdict(int)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout

mode = {'debug': args.debug, 'gpumm': args.gpumm}

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff

ds, vb, ed = load_weibo_ner('./data', unigram_embedding_path='./data/gigaword_chn.all.a2b.uni.ite50.vec',index_token=False)
w_list = extract_word_list('/Users/orange/Desktop/Github/Flat-ner/V1/data/wordsListdemo.txt')
datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(ds, vb, ed, w_list,word_embedding_path=None)

# model path : /Users/orange/.fastNLP/embedding/bert-chinese-wwm
bert_embedding = BertEmbedding(vocabs['lattice'], model_dir_or_name='cn-wwm', requires_grad=False,
                               word_dropout=0.01)


max_seq_len = max(datasets['train']['seq_len'])
# max_seq_len = max(* map(lambda x:max(x['seq_len']), datasets['train']))

# 这里模型配置写死为 BERT + 相对位置编码 + lattice
model = Lattice_Transformer_SeqLabel(embeddings['lattice'],
                                     embeddings['bigram'],
                                     args.hidden,
                                     len(vocabs['label']),
                                     num_heads=args.head,
                                     num_layers=args.layer,
                                     learnable_position=args.learn_pos,
                                     add_position=args.add_pos,
                                     layer_preprocess_sequence=args.pre,
                                     layer_postprocess_sequence=args.post,
                                     ff_size=args.ff,
                                     scaled=args.scaled,
                                     dropout=dropout,
                                     vocabs=vocabs,
                                     max_seq_len=max_seq_len,
                                     rel_pos_shared=args.rel_pos_shared,
                                     k_proj=args.k_proj,
                                     q_proj=args.q_proj,
                                     v_proj=args.v_proj,
                                     r_proj=args.r_proj,
                                     self_supervised=args.self_supervised,
                                     attn_ff=args.attn_ff,
                                     pos_norm=args.pos_norm,
                                     ff_activate=args.ff_activate,
                                     embed_dropout_pos=args.embed_dropout_pos,
                                     four_pos_shared=args.four_pos_shared,
                                     four_pos_fusion=args.four_pos_fusion,
                                     four_pos_fusion_shared=args.four_pos_fusion_shared,
                                     bert_embedding=bert_embedding
                                    )

for k, v in datasets.items():
    v.set_input('lattice','bigrams','seq_len','target')
    v.set_input('lex_num','pos_s','pos_e')
    v.set_target('target','seq_len')
    v.set_pad_val('lattice',vocabs['lattice'].padding_idx)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss = LossInForward()
encoding_type = 'bio'
f1_metric = SpanFPreRecMetric(vocabs['label'],pred='pred',target='target',seq_len='seq_len',encoding_type=encoding_type)
acc_metric = AccuracyMetric(pred='pred',target='target',seq_len='seq_len',)
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric
]

trainer = Trainer(datasets['train'],
                  model,
                  optimizer,
                  loss,
                  args.batch,
                  n_epochs=args.epoch,
                  dev_data=datasets['dev'],
                  metrics=metrics,
                  device=device,
                  dev_batch_size=args.test_batch,
                  test_use_tqdm=False,
                  check_code_level=-1,
                  save_path="./model")  # 保存模型
trainer.train()
