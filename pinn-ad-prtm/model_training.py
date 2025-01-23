#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:lingyuxiang

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from load_dataset import z_score, z_score_with_param, one_std

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi


def create_param(cType):
    """输入特征标准化参数"""
    param = np.zeros((6, 2))

    wn = np.arange(0.4, 1.7+10e-4, 10e-3)
    if cType == 'WC':
        re = np.arange(0., 40+10e-4, 10e-3)
    else:
        re = np.arange(0., 90+10e-4, 10e-3)
    # re = np.arange(0., 90+10e-4, 10e-3)
    tau = 10 ** np.arange(-2., 2+10e-4, 10e-3)
    mu = np.cos(np.arange(0., 90+10e-4, 10e-3) * deg2rad)
    phi = np.arange(0., 180+10e-4, 10e-3)

    val_list = [wn, re, tau, mu, mu, phi]
    for i in range(len(val_list)):
        value = val_list[i]
        param[i, 0] = np.mean(value)
        param[i, 1] = np.std(value)
    return param


def pre_process(x, cType):
    # 数据清洗
    x = x[(x[:, 7] > 0), :]

    # P矩阵预处理
    p = x[:, 8:].copy()
    x[:, 8:] /= x[:, 7:8]
    x[:, 7] = np.log10(x[:, 7])

    # 数据归一化
    param = create_param(cType)
    for i in range(6):
        x[:, i] = z_score_with_param(x[:, i],
                                     param[i, 0], param[i, 1])

    x[:, 7], p1, p2 = z_score(x[:, 7])
    param = np.vstack((param, np.array([p1, p2]).reshape(1, -1)))
    for i in range(15):
        x[:, i+8], p1, p2 = one_std(x[:, i+8])
        p[:, i] /= p2
        param = np.vstack((param, np.array([p1, p2]).reshape(1, -1)))

    ds = np.hstack((x, p))
    return ds, param


def cut_dataset(dataset, rand_seed):
    np.random.seed(rand_seed)
    np.random.shuffle(dataset)
    train = dataset[:int(dataset.shape[0] * 0.8)]
    valid = dataset[int(dataset.shape[0] * 0.8):int(dataset.shape[0] * 0.9)]
    test = dataset[int(dataset.shape[0] * 0.9):]
    return train, valid, test


class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('mape16'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_mape16'))

    def draw_p(self, lists, label, draw_type):
        if not os.path.exists(rf'./train_result/{train_name}'):
            os.mkdir(rf'./train_result/{train_name}')
        lists = np.array(lists)
        plt.figure()
        plt.plot(range(len(lists)), lists, label=label)
        if label == 'loss':
            plt.yscale('log')
        plt.ylabel(label)
        plt.xlabel(draw_type)
        plt.legend(loc='upper right')
        plt.savefig(rf'./train_result/{train_name}/{draw_type}_{label}.jpg')
        np.savetxt(rf'./train_result/{train_name}/{draw_type}_{label}.txt', lists)

    def end_draw(self):
        self.draw_p(self.losses['epoch'], 'loss', 'train')
        self.draw_p(self.accuracy['epoch'], 're', 'train')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val')
        self.draw_p(self.val_acc['epoch'], 're', 'val')


@tf.function
def LeakyRelu(x):
    leak = 0.2
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


class WeightedLoss(keras.losses.Loss):
    def __init__(self, scale, mean, **kwargs):
        super().__init__(name='WeightedLoss', **kwargs)
        self.scale = scale
        self.mean = mean

    def call(self, y_true, y_pred):
        loss_y15 = self.get_loss_y15(y_true[:, 1:16], y_pred[:, 1:])
        loss_I = self.get_loss_I(y_true[:, 0:1], y_pred[:, 0:1])
        loss_P = self.get_loss_P(y_true, y_pred)

        loss = loss_y15 + loss_I + loss_P
        return loss

    def get_loss_y15(self, y_true, y_pred):
        weights = 1 / (.25 + .75 * tf.abs(y_true))
        loss_1 = tf.where(tf.abs(y_pred) < 5e-6,
                          tf.zeros_like(y_pred),
                          tf.abs(y_pred) / .25)
        loss_2 = tf.where(tf.abs(y_pred) < 5e-6,
                          tf.ones_like(y_pred) * 5,
                          weights * tf.abs(y_true - y_pred))
        pred_loss = tf.where(tf.equal(y_true, 0), loss_1, loss_2)
        return tf.reduce_mean(pred_loss, axis=-1)

    def get_loss_I(self, y_true, y_pred):
        weights = 1 / (.1 + .9 * tf.abs(y_true))
        pred_loss = weights * tf.abs(y_true - y_pred) + tf.square(y_true - y_pred)
        return tf.reduce_mean(pred_loss, axis=-1)

    def get_loss_P(self, y_true, y_pred):
        weights = 1 / (.25 + .75 * tf.abs(y_true[:, 16:]))
        P_11 = tf.pow(10., tf.add(tf.multiply(y_pred[:, :1], self.scale), self.mean))
        p_pred = tf.multiply(y_pred[:, 1:], P_11)
        pred_loss = tf.multiply(weights, tf.abs(y_true[:, 16:] - p_pred))
        return tf.reduce_mean(pred_loss, axis=-1)

    def get_config(self):
        return {'scale': self.scale,
                'mean': self.mean}


class MeanAbsolutePercentageError(keras.metrics.Metric):
    def __init__(self, name="mape", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        epsilon = 1e-7
        y_pred = tf.convert_to_tensor(y_pred[:, :1])
        y_true = tf.cast(y_true[:, :1], y_pred.dtype)
        mape = tf.abs(
            (y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon)
        )
        self.total_error.assign_add(tf.reduce_sum(mape))
        self.total_count.assign_add(tf.cast(tf.size(y_pred), tf.float32))

    def result(self):
        return 100 * self.total_error / self.total_count

    def reset_state(self):
        self.total_error.assign(0.0)
        self.total_count.assign(0.0)


class MeanAbsolutePercentageError16(MeanAbsolutePercentageError):
    def __init__(self, name="mape16", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        epsilon = 1e-7
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true[:, :16], y_pred.dtype)
        mape = tf.abs(
            (y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon)
        )
        self.total_error.assign_add(tf.reduce_sum(mape))
        self.total_count.assign_add(tf.cast(tf.size(y_pred), tf.float32))


class PINNModel(keras.models.Model):
    def __init__(self, param):
        super().__init__(name='PINN')

        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(128, activation='relu')
        self.dense3 = keras.layers.Dense(256, activation=LeakyRelu)
        self.dense4 = keras.layers.Dense(128, activation=LeakyRelu)
        self.dense5 = keras.layers.Dense(128, activation=LeakyRelu)
        self.outputs = keras.layers.Dense(16, activation=None)

        self.p_mean = param[0]
        self.p_scale = param[1]

    def call(self, inputs, training=None, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.outputs(x)

    def compile(self, optimizer='adam', metrics=None, **kwargs):
        # tf.keras.optimizers.legacy.Adam
        super().compile(optimizer=optimizer,
                        loss=WeightedLoss(self.p_scale, self.p_mean),
                        metrics=[MeanAbsolutePercentageError16()],
                        **kwargs)

    def get_config(self):
        return {'dense1': self.dense1,
                'dense2': self.dense2,
                'dense3': self.dense3,
                'dense4': self.dense4,
                'dense5': self.dense5,
                'outputs': self.outputs,
                'p_mean': self.p_mean,
                'p_scale': self.p_scale}


if __name__ == '__main__':
    # model name
    train_name = 'T_WC_PINN'
    cType = train_name.split('_')[1]

    # dataset
    dataset = np.loadtxt(rf'./data/{cType}_stokes/T.dat')

    # normalization
    dataset, revParam = pre_process(dataset, cType)
    print(dataset.shape[0])
    # np.savetxt(rf'model_param/rev_param_T_WC.txt', np.array(revParam))

    # model settings
    tf.keras.backend.clear_session()
    model = PINNModel(revParam[6, :])
    model.build((None, 7))
    model.compile()
    model.summary()

    train_dataset, valid_dataset, test_dataset = cut_dataset(dataset, 114)
    x_train = train_dataset[:, :7]
    y_train = train_dataset[:, 7:]
    x_val = valid_dataset[:, :7]
    y_val = valid_dataset[:, 7:]
    x_test = test_dataset[:, :7]
    y_test = test_dataset[:, 7:]
    logs_loss = LossHistory()
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=250,
                        batch_size=512,
                        callbacks=[logs_loss])
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.1f}s')
    model.save(rf'model_param/layer{train_name}')
    logs_loss.end_draw()
