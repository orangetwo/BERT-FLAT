# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/9/26 11:02 上午
# @File    : utils.py
import collections

import torch
# from torch import nn
import torch.nn as nn
from fastNLP.modules import ConditionalRandomField


def get_skip_path(chars, w_trie):
    sentence = ''.join(chars)
    result = w_trie.get_lexicon(sentence)

    return result


def get_skip_path_trivial(chars, w_list):
    chars = ''.join(chars)
    w_set = set(w_list)
    result = []
    # for i in range(len(chars)):
    #     result.append([])
    for i in range(len(chars) - 1):
        for j in range(i + 2, len(chars) + 1):
            if chars[i:j] in w_set:
                result.append([i, j - 1, chars[i:j]])

    return result


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self, w):
        '''
        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i, j, sentence[i:j + 1]])

        return result


def extract_word_list(filename):
    word_list = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:

            tmp = line.strip()
            if tmp:
                word_list.append(tmp)

    return word_list


class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0<=p<=1
        self.p = p

    def forward(self, x):
        if self.training and self.p>0.001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x)
            # print(mask.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0)/(1-self.p)
        return x


def get_crf_zero_init(label_size, include_start_end_trans=False, allowed_transitions=None,
                 initial_method=None):

    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
    return crf


if __name__ == '__main__':
    w_list = ["重庆","人和药店","药店"]
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    print(w_trie.get_lexicon("重庆人和药店"))
