> **Note**
> This repository is no longer actively maintained by Babylon Health. For further assistance, reach out to the paper authors.

# CorrSim

**CorrSim** is an evaluation framework and a collection of statistical similarity measures for word vectors described in

Vitalii Zhelezniak, Aleksandar Savkov, April Shen, and Nils Y. Hammerla. *Correlation Coefficients and Semantic Textual Similarity, NAACL-HLT 2019.*

**CorrSet** is a a collection of multivariate statistical similarity measures for word vectors described in

Vitalii Zhelezniak, April Shen, Daniel Busbridge, Aleksandar Savkov, and Nils Y. Hammerla. *Correlations between Word Vector Sets, EMNLP-IJCNLP 2019.*

**InfoSim** is a collection of mutual information similarity measures for word vectors described in

Vitalii Zhelezniak, Aleksandar Savkov, and Nils Y. Hammerla. *Estimating Mutual Information Between Dense Word Embeddings, ACL 2020.*

## Dependencies

This code is written in Python 3. The requirements are listed in `requirements.txt`.
```
pip3 install -r requirements.txt
```


## Evaluation tasks

The experimental framework derived from [SentEval](https://github.com/facebookresearch/SentEval) evaluates the similarity measures on the following datasets:

| [STS 2012](https://www.cs.york.ac.uk/semeval-2012/task6/)   | [STS 2013](http://ixa2.si.ehu.es/sts/) | [STS 2014](http://alt.qcri.org/semeval2014/task10/) | [STS 2015](http://alt.qcri.org/semeval2015/task2/) | [STS 2016](http://alt.qcri.org/semeval2016/task1/) |

To get all the datasets, run (in data/downstream/):
```bash
./get_sts_data.bash
```
This will automatically download and preprocess the downstream datasets, and store them in data/downstream (warning: for MacOS users, you may have to use p7zip instead of unzip).


## Experiments

Word vectors files must be in a word2vec txt format and are placed in `data/word_vectors/`.
The mapping from word vector model name to filename is found in `evaluation/utils.py`.

```python

WORD_VEC_MAP = {
    'glove': 'glove.840B.300d.w2vformat.txt',
    'word2vec': 'GoogleNews-vectors-negative300.txt',
    'fasttext': 'fasttext-crawl-300d-2M.txt'
}
```

All the experiments are located in `evaluation`. They include

1. `conf_intervals.py` - evaluates rank correlation similarities against cosine similarity on STS and computes 95% BCa confidence intervals for the delta in performance.
2. `corrsim_eval.py` - evaluates rank correlation similarities against cosine similarity on STS and computes fractions of non-normal word and sentence embeddings.
3. `wordsim_eval.py` - evaluates rank correlation similarities against cosine similarity and Pearson correlation on word-level tasks.
3. `corrset_eval.py` - evaluates multivariate correlations on STS (EMNLP-IJCNLP 2019).
4. `infosim_eval.py` - evaluates the KSG estimator of mutual information on STS (ACL 2020).

## Feedback and Contact:

If this code is useful to your research, please consider citing

Vitalii Zhelezniak, Aleksandar Savkov, April Shen, and Nils Y. Hammerla. *Correlation Coefficients and Semantic Textual Similarity, NAACL-HLT 2019.*

and/or

Vitalii Zhelezniak, April Shen, Daniel Busbridge, Aleksandar Savkov, and Nils Y. Hammerla. *Correlations between Word Vector Sets, EMNLP-IJCNLP 2019.*

and/or

Vitalii Zhelezniak, Aleksandar Savkov, and Nils Y. Hammerla. *Estimating Mutual Information Between Dense Word Embeddings, ACL 2020.*

Contact: Vitalii Zhelezniak <vitali.zhelezniak@babylonhealth.com>

## Related work
* [E. Santus, H. Wang, E. Chersoni, Y. Zhang - A rank-based similarity metric  for  word  embeddings, ACL 2018](https://www.aclweb.org/anthology/P18-2088)
* [V. Zhelezniak, A. Savkov, A. Shen, F. Moramarco, J. Flann, N. Y. Hammerla - Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors, ICLR 2019](https://openreview.net/pdf?id=SkxXg2C5FX)
