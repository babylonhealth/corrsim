# Copyright 2018 Babylon Partners Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This source code is derived from SentEval source code.
# SentEval Copyright (c) 2017-present, Facebook, Inc.
# ==============================================================================

from __future__ import absolute_import, division, unicode_literals

import io
import numpy as np
import logging
from scipy.stats import shapiro


WORD_VEC_MAP = {
    'glove': 'glove.840B.300d.w2vformat.txt',
    'word2vec': 'GoogleNews-vectors-negative300.txt',
    'fasttext': 'fasttext-crawl-300d-2M.txt',
}


def get_word_vec_path_by_name(word_vec_name):
    base_path = '../data/word_vectors/'
    return base_path + WORD_VEC_MAP[word_vec_name]


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id,
                norm=False):
    """
    Loads words and word vectors from a text file
    :param path_to_vec: path to word vector file in word2vec format
    :param word2id: words to load
    :param norm: normalise word vectors
    :return: dict containing word: word vector
    """
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # always skip the first line, contains num of words and dim
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                np_vector = np.fromstring(vec, sep=' ')
                if norm:
                    np_vector = np_vector / np.linalg.norm(np_vector)
                word_vec[word] = np_vector

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    print_frac_normal(word_vec.values())
    return word_vec


def print_frac_normal(word_vectors):
    reject_normal = sum([1 for wv in word_vectors if shapiro(wv)[1] < 0.05])
    logging.debug('Fraction of words where normality was not rejected = %.4f'
                  % (1 - 1.0 * reject_normal / len(word_vectors)))
