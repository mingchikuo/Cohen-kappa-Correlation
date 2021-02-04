from __future__ import print_function
# -----------------------------------------
#   Author
# -----------------------------------------
"""
<summary>
Computing Cohen kappa correlation of two ordered lists or arrays.
</summary>

<file>
kappa.py
</file>

<class>
Script
</class>

<author>
Author:
Mitchel Kuo (Mitchel.Kuo@acer.com)
            (mingchikuo7@gmail.com)
</author>

<modifier>
Modifier:
Mitchel Kuo (Mitchel.Kuo@acer.com)
            (mingchikuo7@gmail.com)
</modifier>

<remarks> </remarks>
<copyright> </copyright>
"""

# -----------------------------------------
#   Specifications
# -----------------------------------------
"""
----------
Parameters
----------
y1 : array, shape = [n_samples]
Labels assigned by the first annotator.

y2 : array, shape = [n_samples]
Labels assigned by the second annotator. The kappa statistic is
symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.

labels : array, shape = [n_classes], optional
List of labels to index the matrix. This may be used to select a
subset of labels. If None, all labels that appear at least once in
``y1`` or ``y2`` are used.

weights : str, optional
Weighting type to calculate the score. None means no weighted;
"linear" means linear weighted; "quadratic" means quadratic weighted.

sample_weight : array-like of shape (n_samples,), default=None
Sample weights.

-------
Returns
-------
kappa : float
The kappa statistic, which is a number between -1 and 1. The maximum
value means complete agreement; zero or lower means chance agreement.

----------
References
----------
 `Wikipedia entry for the Cohen's kappa.
            <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_
"""

# -----------------------------------------
#   Packages
# -----------------------------------------
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import os
import json
from lib.utils import *

def cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None):
    confusion = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k

# -----------------------------------------
#   Do Your Code Here
# -----------------------------------------
def main():
    # Swapping y1 & y2 is Ok
    y1 = [ 1, 0, 1, 2, 0]
    y2 = [ 1, 1, 1, 1, 0]

    # There is the same result from 'linear' or 'quadratic' in Binary classification
    kappa = round(cohen_kappa_score(y1, y2, weights='linear'), 5)
    print('Cohen Kappa係數:', kappa)

if __name__ == '__main__':
    main()
