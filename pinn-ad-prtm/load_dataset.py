#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:lingyuxiang

import os
import numpy as np

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi


def z_score(x):
    # change x into zero-mean normalization
    if np.min(x) == np.max(x):
        return np.zeros_like(x), x[0], 1

    xMean = np.mean(x)
    xStd = np.std(x)
    return (x - xMean) / xStd, xMean, xStd


def one_std(x):
    # change std(x) to 1
    if np.min(x) == np.max(x):
        return np.zeros_like(x), x[0], 1

    xStd = np.std(x)
    return x / xStd, 0, xStd


def z_score_with_param(x, xMean, xStd):
    # change x into zero-mean normalization
    return (x - xMean) / xStd


def un_z_score(x, xMean, xStd):
    # return x from zero-mean normalization
    return x * xStd + xMean


if __name__ == '__main__':
    pass
