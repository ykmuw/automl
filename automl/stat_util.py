# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats

def smirnov_grubbs(data, alpha):
    """
    Find outliers by Grubbs's test (https://en.wikipedia.org/wiki/Grubbs%27s_test)

    Parameters
    ----------
    data : numpy.array
        data array for test
    alpha : float
        significance level

    Returns
    -------
    outliers : numpy.array
        outliers
    """

    x, o = list(data), []
    while True:
        n = len(x)
        t = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
        tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
        i_min, i_max = np.argmin(x), np.argmax(x)
        myu, std = np.mean(x), np.std(x, ddof=1)
        i_far = i_max if np.abs(
            x[i_max] -
            myu) > np.abs(
            x[i_min] -
            myu) else i_min
        tau_far = np.abs((x[i_far] - myu) / std)
        if tau_far < tau:
            break
        o.append(x.pop(i_far))
    return np.array(o)
