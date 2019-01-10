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

from glob import glob
import io
from itertools import combinations
import logging
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata, shapiro
import scikits.bootstrap as bstrap

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/')
vectors_dir = os.path.join(data_path, "word_vectors")
tasks_dir = os.path.join(data_path, "word-sim/")

vector_map = {
    'GloVe': f'{vectors_dir}/glove.840B.300d.w2vformat.txt',
    'fastText': f'{vectors_dir}/fasttext-crawl-300d-2M.txt',
    'word2vec': f'{vectors_dir}/GoogleNews-vectors-negative300.txt',
}

SHAPIRO_THRESHOLD = 0.05
EPS = np.finfo(float).eps


def statistic(data):
    gs = data[:, 0]
    sys_a = data[:, 1]
    sys_b = data[:, 2]
    r1 = spearmanr(gs, sys_a)[0]
    r2 = spearmanr(gs, sys_b)[0]
    return r1 - r2


def apsynp(u, v):
    ur = rankdata(u)
    vr = rankdata(v)
    ur = np.power(ur, 0.1)
    vr = np.power(vr, 0.1)
    avgr = np.mean([ur, vr], axis=0)
    return np.sum(1 / avgr), 0


def cosine(u, v):
    return np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)), None


sim_map = {
    'COS': cosine,
    'PRS': pearsonr,
    'SPR': spearmanr,
    'KEN': lambda u, v: kendalltau(u, v, method='asymptotic'),
    'APS': apsynp
}

ranked_sims = ['SPR', 'KEN']
other_sims = ['COS', 'PRS']


def get_wordvec(path_to_vec, word2id):
    word_vec = {}
    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


def load_file(fp):
    df = pd.read_csv(fp, sep='\t', names=['word1', 'word2', 'human_score'])
    df['dataset'] = os.path.basename(fp)[3:-4]
    return df


def load_dataset(data_dir):
    global_df = None
    files = glob(f'{data_dir}*')
    for fp in files:
        df = load_file(fp)
        if global_df is None:
            global_df = df
        else:
            global_df = global_df.append(df)
    return global_df[['word1', 'word2', 'dataset', 'human_score']]


def apply_sim(sim_func, vex, row):
    w1 = row['word1']
    w2 = row['word2']
    try:
        u = vex[w1]
        v = vex[w2]
        sim, _ = sim_func(u, v)
        return sim
    except KeyError:
        return None


def calculate_similarities(df, vex, sims, exclude=()):
    for sim in sims:
        print(f'Calculating {sim}')
        sim_func = sim_map[sim]
        df[sim] = df.apply(
            lambda x:
            apply_sim(sim_func, vex, x)
            if x['dataset'] not in exclude
            else None,
            axis=1)


def calculate_shapiro(corrs, df, vex, exclude=()):
    for ds in set(df['dataset']):
        if ds in exclude:
            continue
        df_ds = df[df['dataset'] == ds]
        words = set(np.unique(df_ds[['word1', 'word2']].values)) & set(vex)
        non_norms = 0
        for w in words:
            _, p = shapiro(vex[w])
            if p < SHAPIRO_THRESHOLD:
                non_norms += 1
        corrs[ds]['shapiro'] = non_norms / (len(words) + EPS)


def compute_ci(scores, exclude=()):
    cfs = {}
    for ds, items in scores.items():
        if ds in exclude:
            continue
        cfs[ds] = {}
        for sim1, sim2 in combinations(items, 2):
            print(f'Computing CI for {ds} - {sim1} : {sim2}')
            human_scores, sim_scores1 = scores[ds][sim1]
            _, sim_scores2 = scores[ds][sim2]
            data = list(zip(human_scores, sim_scores1, sim_scores2))
            cfs[ds][(sim1, sim2)] = \
                bstrap.ci(data, statfunction=statistic, method='bca')
    return cfs


def correlation(df, sims, exclude=()):
    corrs = {}
    scores = {}
    datasets = set(df['dataset'])
    for ds in datasets:
        if ds in exclude:
            continue
        corrs[ds] = {}
        scores[ds] = {}
        for sim in sims:
            sim_scores = df[(df['dataset'] == ds) &
                            (df[sim].notnull())][sim]
            human_scores = df[(df['dataset'] == ds) &
                              (df[sim].notnull())
                              ]['human_score']
            corrs[ds][sim], _ = spearmanr(human_scores, sim_scores)
            scores[ds][sim] = (human_scores, sim_scores)
        corrs[ds]['size'] = df[df['dataset'] == ds].shape[0]
    return corrs, scores


def vector_correlations(df, word_set, vex_fp, sims, exclude=()):
    vectors = get_wordvec(vex_fp, word_set)
    calculate_similarities(df, vectors, sims, exclude=exclude)
    corrs, scores = correlation(df, sims, exclude=exclude)
    cfs = compute_ci(scores, exclude=exclude)
    calculate_shapiro(corrs, df, vectors, exclude=exclude)
    return corrs, cfs


def max_idxs(lst):
    max_idxs = [0]
    max_ = -10000
    for i, itm in enumerate(lst):
        if float(itm) > max_:
            max_idxs = [i]
            max_ = float(itm)
        elif float(itm) == max_:
            max_idxs.append(i)
    return max_idxs


def max_ci_idxs(corrs, scores):
    max_names = [0]
    max_ = -10000
    name_lst = list(corrs)
    for name, val in corrs.items():
        if float(val) > max_:
            max_names = [name]
            max_ = float(val)
        elif float(val) == max_:
            max_names.append(name)
    name = max_names[0]
    sims = [sim2 for sim1, sim2 in scores if sim1 == name]
    for sim in sims:
        a, b = scores[(name, sim)]
        if a < 0 < b:
            max_names.append(sim)
    return [name_lst.index(name) for name in max_names]


def get_winner(corrs, scores):
    max_nonrank = 'COS' if corrs['COS'] > corrs['PRS'] else 'PRS'
    max_rank = 'SPR' if corrs['SPR'] > corrs['KEN'] else 'KEN'
    try:
        a, b = scores[(max_rank, max_nonrank)]
    except KeyError:
        a, b = scores[(max_nonrank, max_rank)]
    not_significant = a < 0 < b
    if not_significant:
        return '='
    elif corrs[max_rank] < corrs[max_nonrank]:
        return 'N'
    else:
        return 'R'


def to_table(corrs, sims, exclude=()):
    columns = ['', 'task', 'r', *sims]
    table = [
        '\\begin{table}',
        '\\begin{tabular}{c@{\hskip 12pt}kssssss}',
        '\\toprule',
        ' & '.join([f'\\textbf{{{c}}}' for c in columns]) + ' \\\\',
        '\\midrule',
        '\\toprule'
    ]
    for vec_name, vec_path in vector_map.items():
        tasks = [t for t in set(corrs[vec_name].keys()) if t not in exclude]
        head_cell = f'\\textbf{{\\multirow{{{len(tasks)}}}{{*}}' \
                    f'{{\\rotatebox[origin=c]{{90}}{{{vec_name}}}}}}}'
        for task in tasks:
            values = ['%.1f' % (corrs[vec_name][task][sim] * 100)
                      for sim in sims]
            ratio = ('%d' % (corrs[vec_name][task]['shapiro'] * 100)) + '\\%'
            for idx in max_idxs(values):
                values[idx] = f'\\textbf{{{values[idx]}}}'
            row = [head_cell, task, ratio, *values]
            head_cell = ''
            table.append(' & '.join(row) + ' \\\\')
        table.append('\\toprule')
    table.append('\\end{tabular}')
    table.append('\\caption{Some caption}')
    table.append('\\label{tab:word-sim}')
    table.append('\\end{table}')
    return '\n'.join(table)


def to_ci_table(corrs, scores, sims, exclude=()):
    columns = ['', 'task', 'N', 'V', *sims]
    table = [
        '\\begin{table}',
        '\\begin{tabular}{s@{\hskip 7pt}kj@{\hskip 8pt}s@{\hskip 8pt}ss'
        '@{\hskip 8pt}ss}',
        '\\toprule',
        ' & '.join([f'\\textbf{{{c}}}' for c in columns]) + ' \\\\',
        '\\midrule',
        '\\toprule'
    ]
    for vec_name, vec_path in vector_map.items():
        tasks = [t for t in set(corrs[vec_name].keys()) if t not in exclude]
        head_cell = f'\\textbf{{\\multirow{{{len(tasks)}}}{{*}}' \
                    f'{{\\rotatebox[origin=c]{{90}}{{{vec_name}}}}}}}'
        for task in tasks:
            task_corrs = corrs[vec_name][task]
            task_scores = scores[vec_name][task]
            values = ['%.1f' % (task_corrs[sim] * 100) for sim in sims]
            ratio = ('%.2f' % (1 - task_corrs['shapiro']))
            winner = get_winner(task_corrs, task_scores)
            row = [
                head_cell, f'\\textsc{{{task.lower()}}}', ratio[1:], winner,
                *values
            ]
            head_cell = ''
            table.append(' & '.join(row) + ' \\\\')
        table.append('\\toprule')
    table.append('\\end{tabular}')
    table.append('\\caption{Some caption}')
    table.append('\\label{tab:word-sim}')
    table.append('\\end{table}')
    return '\n'.join(table)


if __name__ == '__main__':
    excluded = [
        'MEN-TR-3k', 'MTurk-771', 'WS-353-REL', 'WS-353-ALL', 'YP-130'
    ]
    sims = ['COS', 'PRS', 'SPR', 'KEN']
    df = load_dataset(tasks_dir)
    word_set = set(df['word1']) | set(df['word2'])
    corrs = {}
    cis = {}
    for vex, vex_fp in vector_map.items():
        print(f'======= Vectors: {vex} =======')
        corrs[vex], cis[vex] = vector_correlations(df, word_set, vex_fp, sims,
                                                   exclude=excluded)
        print('================================')
    print(to_table(corrs, sims, exclude=excluded))
    print(to_ci_table(corrs, cis, sims, exclude=excluded))
