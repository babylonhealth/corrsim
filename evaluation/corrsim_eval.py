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

import sys
import numpy as np
import logging
import itertools

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'


sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import utils
from similarity import get_similarity_by_name

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def prepare(params, samples):
    word_vec_path = utils.get_word_vec_path_by_name(params.word_vec_name)
    params.wvec_dim = 300

    _, params.word2id = utils.create_dictionary(samples)
    params.word_vec = utils.get_wordvec(word_vec_path,
                                        params.word2id)
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        embeddings.append(sentvec)

    return embeddings


if __name__ == "__main__":
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

    word_vectors = [
        'glove',
        'fasttext',
        'word2vec',
    ]

    similarities = [
        'avg_cosine',
        'pearson',
        'spearman',
        'kendall',
        'apsyn',
        'apsynp',
    ]

    results = []
    experiments = list(itertools.product(word_vectors,
                                         similarities))
    logging.info('Running {0} experiments. Good luck! :)\n\n\n'.format(len(experiments)))

    for idx, experiment in enumerate(experiments):
        word_vec_name = experiment[0]
        sim_name = experiment[1]

        logging.info('Word vectors: {0}'.format(word_vec_name))
        logging.info('Similarity: {0}'.format(sim_name))
        logging.info('BEGIN\n\n\n')

        params_senteval = {
            'task_path': PATH_TO_DATA
        }
        params_experiment = {
            'word_vec_name': word_vec_name,
            'similarity_name': sim_name
        }
        params_senteval.update(params_experiment)
        params_senteval['similarity'] = get_similarity_by_name(
            sim_name)

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        result = se.eval(transfer_tasks)
        result_dict = {
            'param': params_experiment,
            'eval': result
        }
        results.append(result_dict)
        logging.info('END. Experiment #{0} saved\n\n\n'.format(idx + 1))
