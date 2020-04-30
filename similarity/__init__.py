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

from .baseline import *
from .correlation import *
from .cka import *
from .mi import *

NAME_TO_SIM = {
    # CorrSim
    'avg_cosine': avg_cosine,
    'pearson': pearson,
    'spearman': spearman,
    'kendall': kendall,
    'max_spearman': max_spearman,

    'apsyn': apsyn,
    'apsynp': apsynp,

    # CorrSet
    'cka_linear': cka_factory(linear_kernel),
    'cka_gaussian': cka_factory(gaussian_kernel),
    'cka_dcorr': dcorr,

    # InfoSim
    'ksg3': ksg3,
    'ksg10': ksg10,
    'mean_ksg10': mean_ksg10,
    'max_ksg10': max_ksg10
}


def get_similarity_by_name(sim_name):
    return NAME_TO_SIM[sim_name]
