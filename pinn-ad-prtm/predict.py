#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:lingyuxiang

import os
import sys
import numpy as np
from keras import models
from keras.utils import custom_object_scope
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.transforms as mtransforms
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde, pearsonr

from model_training import *
from load_dataset import z_score_with_param, un_z_score


srcDir = os.path.dirname(__file__)
sys.path.append(srcDir)
modelDir = os.path.join(srcDir, 'model_param')


def loadFile(infile):
    """
    :param infile:
    :return: ds: header + p
                header: wavenumber, re, tau, mu_s, mu_v, phi(deg)
                p: 4 stokes matrix
             theta_v(deg)
             phi(rad)
    """
    pType = infile.split('/')[-1].split('_')[0]
    ds = np.loadtxt(infile, skiprows=4)
    p_ds = ds[:, 3:]
    header = np.empty((ds.shape[0], 7))

    with open(infile, 'r') as f:
        wn = float(f.readline().split('=')[-1][:-3])
        re = float(f.readline().split('=')[-1][:-3])
        tau = float(f.readline().split('=')[-1])
    sza = ds[:, 0]
    vza = ds[:, 1]
    if pType == 'T':
        vza = 180 - vza
    raa = ds[:, 2]
    header[:, 0] = wn
    header[:, 1] = re
    header[:, 2] = tau
    header[:, 3] = np.cos(sza * deg2rad)
    header[:, 4] = np.cos(vza * deg2rad)
    header[:, 5] = raa  # degree exactly
    header[:, 6] = \
        np.sin(sza * deg2rad) * np.sin(vza * deg2rad) * np.cos(raa * deg2rad) + \
        np.cos(sza * deg2rad) * np.cos(vza * deg2rad)
    header[header[:, 6] > 1, 6] = 1.
    header[header[:, 6] < -1, 6] = -1.
    total_ds = np.hstack((header, p_ds))
    theta_v = vza.reshape((90, 37)).squeeze()
    phi = raa.reshape((90, 37)).squeeze()
    phi = phi * deg2rad
    return total_ds, theta_v, phi


def pre_process(x, param):
    for i in range(6):
        x[:, i] = z_score_with_param(x[:, i], param[i, 0], param[i, 1])
    return x


def post_process(x, y, param):
    # sep
    for i in range(16):
        y[:, i] = un_z_score(y[:, i], param[i+6, 0], param[i+6, 1])

    y[:, 0] = 10 ** y[:, 0]
    y[:, 1:] *= y[:, 0:1]
    return y


def predict_metrix(stokesType:str, particleType:str, header:np.ndarray):
    # load models
    if stokesType == 'R':
        if particleType == 'TH':
            modelName = 'R_TH_PINN'
            stokesType = 'R_TH'
        elif particleType == 'WC':
            modelName = 'R_WC_PINN'
            stokesType = 'R_WC'
    else:
        if particleType == 'TH':
            modelName = 'T_TH_PINN'
            stokesType = 'T_TH'
        elif particleType == 'WC':
            modelName = 'T_WC_PINN'
            stokesType = 'T_WC'

    # get reverse parameters
    revParam = np.loadtxt(os.path.join(modelDir, rf'rev_param_{stokesType}.txt'))

    # load model
    with custom_object_scope({'WeightedLoss': WeightedLoss(scale=revParam[6, 1], mean=revParam[6, 0]),
                              'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
                              'MeanAbsolutePercentageError16': MeanAbsolutePercentageError16()}):
        model = tf.keras.saving.load_model(os.path.join(modelDir, rf'layer{modelName}'))

    # get input features
    inputFtrs = pre_process(header.copy(), revParam)

    # predict outputs
    predict = model.predict(inputFtrs, verbose=0)
    predict = post_process(inputFtrs, predict, revParam)
    return predict


def polar_plot(x1, x2, y):
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5),
                           subplot_kw={'projection': 'polar'},
                           gridspec_kw={'wspace': 0.3, 'bottom': 0.1})
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-.1, vcenter=0, vmax=.1)
    for i in range(4):
        ax[i].set_xlim(0, np.pi)
        im = ax[i].contourf(x1, x2,
                               y[..., i],
                               levels=50,
                               cmap=cmap)

        ax[i].set_theta_zero_location('W')
        ax[i].set_theta_direction(-1)
        ax[i].set_yticks([10 * n for n in range(1, 9)])
        ax[i].set_yticklabels([])
        ax[i].tick_params(labelsize=8, pad=0)
    plt.show()


def scatter_compare(v1, v2, draw_type=2, draw_idx=0):
    """
    :param draw_type: 1: single  2: 16 elements
    :param draw_idx: idx of element (valid only when draw_type==1)
    """
    min_list = [0, -.04, -.03, -.0003]
    max_list = [1.5, .04, .03, .0003]

    if draw_type == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        tmp_R_real = v1[..., draw_idx].flatten()
        tmp_R_pred = v2[..., draw_idx].flatten()

        xy = np.vstack([tmp_R_real, tmp_R_pred])
        z = gaussian_kde(xy)(xy)
        ax.scatter(tmp_R_real, tmp_R_pred, c=z, cmap='jet', s=1.2)
        ax.set_xlabel('Adding-doubling result', labelpad=15)
        ax.set_ylabel('PINN predict')

        r2 = pearsonr(tmp_R_real, tmp_R_pred)[0] ** 2
        bias = np.abs(tmp_R_pred - tmp_R_real)
        re = (tmp_R_pred / tmp_R_real - 1) * 100
        mae = np.nanmean(bias)
        mre = np.nanmean(np.abs(re[~np.isinf(re)]))
        rmse = np.sqrt(np.nanmean(bias ** 2))
        tick_formatter = ScalarFormatter(useMathText=True)
        tick_formatter.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(tick_formatter)
        ax.yaxis.set_major_formatter(tick_formatter)
        ax.set_xlim(min_list[draw_idx], max_list[draw_idx])
        ax.set_ylim(min_list[draw_idx], max_list[draw_idx])
        ax.plot((-100, 100), (-100, 100), linewidth=0.4, c='k')
        xMinMax = max_list[draw_idx] - min_list[draw_idx]
        textX = min_list[draw_idx] + xMinMax * 0.07
        textY = min_list[draw_idx] + xMinMax * 0.8
        ax.text(textX, textY, f'$R^2$ = {r2:.2f}\n'
                              f'RMSE = {rmse:.2e}\n'
                              f'MRE = {mre:.2f}%')
        print(f'MRE = {mre:.5f}%')

    else:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4),
                               gridspec_kw={'wspace':0.3,
                                            'bottom':0.2})
        for i in range(4):
            tmp_R_real = v1[..., i].flatten()
            tmp_R_pred = v2[..., i].flatten()

            xy = np.vstack([tmp_R_real, tmp_R_pred])
            z = gaussian_kde(xy)(xy)
            ax[i].scatter(tmp_R_real, tmp_R_pred, c=z, cmap='jet', s=1.2)
            ax[i].set_xlabel('ADM result', labelpad=15)
            if i == 0:
                ax[i].set_ylabel('PINN prediction')
            r2 = pearsonr(tmp_R_real, tmp_R_pred)[0] ** 2
            bias = np.abs(tmp_R_pred - tmp_R_real)
            re = (tmp_R_pred / tmp_R_real - 1) * 100
            mae = np.nanmean(bias)
            mre = np.nanmean(np.abs(re[~np.isinf(re)]))
            rmse = np.sqrt(np.nanmean(bias ** 2))
            nrmse = rmse / (np.nanmax(tmp_R_real) - np.nanmin(tmp_R_real))
            tick_formatter = ScalarFormatter(useMathText=True)
            tick_formatter.set_powerlimits((-2, 2))
            ax[i].xaxis.set_major_formatter(tick_formatter)
            ax[i].yaxis.set_major_formatter(tick_formatter)
            ax[i].set_xlim(min_list[i], max_list[i])
            ax[i].set_ylim(min_list[i], max_list[i])
            ax[i].plot((-100, 100), (-100, 100), linewidth=0.4, c='k')
            xMinMax = max_list[i] - min_list[i]
            textX = min_list[i] + xMinMax * 0.07
            textY = min_list[i] + xMinMax * 0.8
            ax[i].text(textX, textY, f'$R^2$ = {r2:.2f}\n'
                                     f'RMSE = {rmse:.2e}')
            textX = min_list[i] + xMinMax * 0.9
            textY = min_list[i] + xMinMax * 0.07
            ax[i].text(textX, textY,
                          r'(%s)' % (chr(ord('a')+i)),
                          **font2)
            print(f'NRMSE_{i+1}: {nrmse * 100:.2f}%')
    plt.show()


font_title = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 15,
              }
font1 = {'family': 'Arial',
         'weight': 'bold',
         'size': 12,
         }
font2 = {'family': 'Arial',
         'size': 12,
         }


if __name__ == '__main__':
    cloudType = 'WC'
    cloudTypeName = 'water'
    pType = 'T'
    ds, theta_v, phi = loadFile(rf'./data/{cloudType}_stokes/{pType}_{cloudTypeName}_ADM.txt')
    features = ds[:, :7]
    target = ds[:, 7:]
    target = target.reshape((90, 37, -1))

    p_value = predict_metrix(pType, cloudType, features)
    p_value = p_value.reshape((90, 37, -1))
    iquv_loc = np.array([0, 4, 8, 12])
    p_value = p_value[..., iquv_loc]

    polar_plot(phi, theta_v, p_value)
    scatter_compare(target, p_value)
