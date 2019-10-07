# Copyright 2019 Babylon Partners Limited. All Rights Reserved.
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

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata


def pearson(x, y):
    """
    Pearson correlation coefficient between two sentences
    represented as averaged word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity measure between the two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    return pearsonr(x_mean, y_mean)[0]


def spearman(x, y):
    """
    Spearman correlation coefficient between two sentences
    represented as averaged word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity measure between the two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    return spearmanr(x_mean, y_mean)[0]


def kendall(x, y):
    """
    Kendall correlation coefficient between two sentences
    represented as averaged word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity measure between the two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    return kendalltau(x_mean, y_mean, method='asymptotic')[0]


def _apsynp(x, y, p):
    """
    APSynP similarity measure (Santus et al., 2018)
    applied to sentences represented as averaged word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :param p: exponents for the ranks. APSyn: p=1, APSynP: p=0.1
    :return: similarity measure between the two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x_ranks = rankdata(x_mean)
    y_ranks = rankdata(y_mean)
    x_ranks_p = np.power(x_ranks, p)
    y_ranks_p = np.power(y_ranks, p)

    avgr = np.mean([x_ranks_p, y_ranks_p], axis=0)
    return np.sum(1 / avgr)


def apsyn(x, y):
    return _apsynp(x, y, 1.0)


def apsynp(x, y):
    return _apsynp(x, y, 0.1)


def max_spearman(x, y):
    """
    Spearman correlation coefficient between two sentences
    represented as max-pooled word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity measure between the two sentences
    """
    x_max = np.max(x, axis=0)
    y_max = np.max(y, axis=0)
    return spearmanr(x_max, y_max)[0]
