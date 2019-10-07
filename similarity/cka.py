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
import dcor


def linear_kernel(X):
    """
    Computes a linear kernel for X
    :param X: word embedding matrix X with shape (k x D)
    :return: linear kernel for X
    """
    return np.dot(X.T, X)


def gaussian_kernel(X, sigma=None):
    """
    Computes a Gaussian kernel for X
    :param X: word embedding matrix X with shape (k x D)
    :param sigma: standard deviation of the kernel
    :return: Gaussian kernel for X
    """
    G = linear_kernel(X)
    SN = np.diag(G)
    SD = -2 * G + SN[:, np.newaxis] + SN[np.newaxis, :]
    if sigma is None:
        sigma = np.sqrt(np.median(SD[SD != 0]))
    K = np.exp(-SD / (2 * sigma ** 2))
    return K


def centering_matrix(d):
    """
    Returns a centering matrix of dimension d
    :param d: dimension of the matrix
    :return: centering matrix of dimension d
    """
    return np.eye(d) - np.ones((d, d)) / d


def cka_factory(kernel=None):
    """
    Builds a Centered Kernel Alignment (CKA) similarity function
    with the specified kernel
    :param kernel: kernel function for CKA
    :return: CKA similarity function
    """
    def hsic(X, Y):
        """
        Computes Hilbert-Schmidt independence criterion (HSIC)
        between word embedding matrices X and Y
        :param X: word embedding matrix X with shape (k x D)
        :param Y: word embedding matrix Y with shape (l x D)
        :return: HSIC (unnormalised) between X and Y
        """
        assert X.shape[1] == Y.shape[1]
        d = X.shape[1]
        H = centering_matrix(d)
        KX = kernel(X)
        KY = kernel(Y)
        return np.trace(KX @ H @ KY @ H)

    def cka(X, Y):
        """
        Computes Centered Kernel Alignment (CKA)
        between word embedding matrices X and Y
        :param X: word embedding matrix X with shape (k x D)
        :param Y: word embedding matrix Y with shape (l x D)
        :return: CKA between X and Y
        """
        return hsic(X, Y) / np.sqrt(hsic(X, X) * hsic(Y, Y))

    return cka


def dcorr(X, Y):
    """
    Computes Distance Correlation (dCorr)
    between word embedding matrices X and Y
    :param x: X: word embedding matrix X with shape (k x D)
    :param y: Y: word embedding matrix Y with shape (l x D)
    :return: distance correlation between X and Y
    """
    return dcor.distance_correlation(X.T, Y.T)
