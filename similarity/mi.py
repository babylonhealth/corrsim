# Copyright 2020 Babylon Partners Limited. All Rights Reserved.
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
from npeet.entropy_estimators import mi


def ksg_factory(k=3, pool=None):
    """
    :param k: number of nearest neighbours
    :param pool: optional pooling function (e.g. np.mean or np.max)
    :return: KSG similarity with given k and pooling (if any)
    """

    def ksg(x, y):
        """
        Kraskov–Stogbauer–Grassberger (KSG) estimator of mutual information
        between two sentences represented as word embedding matrices x and y
        :param x: list of word embeddings for the first sentence
        :param y: list of word embeddings for the second sentence
        :return: KSG similarity measure between the two sentences
        """

        if pool is None:
            xT = x.T
            yT = y.T
        else:
            xT = pool(x, axis=0).reshape(-1, 1)
            yT = pool(y, axis=0).reshape(-1, 1)

        return mi(xT, yT, base=np.e, k=k)
    return ksg


ksg3 = ksg_factory(k=3)
ksg10 = ksg_factory(k=10)

mean_ksg10 = ksg_factory(k=10, pool=np.mean)
max_ksg10 = ksg_factory(k=10, pool=np.max)
