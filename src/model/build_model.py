#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@Time    : 2020/3/5 19:59
@Author  : zz
@File    : build_model.py
@Software: PyCharm
@Desc    : 构建DSSM模型
"""

import sys
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses


def build_deep_layers(net, params, mode, name=''):
    """
    Args:
        net: 输入层结构
        params: 网络相关参数
        mode: 训练模式【trian, eval, pred】
        name: 网络结构名字

    Returns:
        返回构建完成的NN网络
    """
    for i in range(len(params['hidden_units'][0:2])):
        num_hidden_units = params['hidden_units'][i]
        keep_prob = 1.0
        if mode == tf.estimator.ModeKeys.TRAIN:
            keep_prob = 0.7
        net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              name=name + '_hidden_units_{}'.format(
                                  str(num_hidden_units) + '_' + str(i)))
        net = tf.layers.dropout(net, keep_prob)
    num_hidden_units = params['hidden_units'][-1]
    net = tf.layers.dense(net, units=num_hidden_units, activation=None,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          name=name + '_hidden_units_{}'.format(str(num_hidden_units) + '_linear'))
    return net


def _dssm_model(features, labels, mode, params):
    """
    Returns:
        构建双塔模型
    """
    user_emb = tf.feature_column.input_layer(features, params['feature_columns']['user_columns'])
    good_emb = tf.feature_column.input_layer(features, params['feature_columns']['good_columns'])

    with tf.name_scope('user'):
        user_emb = build_deep_layers(user_emb, params, mode, name='user')
    with tf.name_scope('goods'):
        good_emb = build_deep_layers(good_emb, params, mode, name='good')
    
    head = head_lib._binary_logistic_or_multi_class_head(
            n_classes=2, weight_column=None, label_vocabulary=None,
            loss_reduction=losses.Reduction.SUM)
    logits = tf.layers.dense(tf.multiply(user_emb, good_emb), units=head.logits_dimension,
        kernel_initializer=tf.glorot_uniform_initializer())
    preds = tf.sigmoid(logits)

    # similarity = tf.reduce_sum(tf.multiply(user_emb, good_emb), axis=-1)
    # predictions = tf.nn.sigmoid(similarity)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': preds
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels['ctr_label'], tf.float32),
                                                                 logits=tf.cast(logits, tf.float32)))

    auc = tf.metrics.auc(labels['ctr_label'], preds)
    metrics = {'auc': auc}
    tf.summary.scalar('auc', auc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    statrt_learning_rate = params['learning_rate']
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate=statrt_learning_rate,
        global_step=global_step,
        decay_steps=params['decay_steps'],
        decay_rate=params['decay_rate'],
        staircase=False)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def _build_model_function(model_dir, config, feature_columns):
    # 配置运行环境
    run_config = tf.estimator.RunConfig(**config.runconfig).replace(
        session_config=tf.ConfigProto(
            device_count={'GPU': 0, 'CPU': 2},
            inter_op_parallelism_threads=36,
            intra_op_parallelism_threads=36,
            allow_soft_placement=True))

    # 根据要求的模型类型返回对应的estimator
    return tf.estimator.Estimator(
        model_fn=_dssm_model,
        model_dir=model_dir,
        params={
            'feature_columns': feature_columns,
            'decay_rate': config.model['dnn_decay_rate'],
            'decay_steps': config.model['dnn_decay_steps'],
            'hidden_units': config.model['dnn_hidden_units'],
            'learning_rate': config.model['dnn_initial_learning_rate'],
            'feature_config': config.read_feature_conf(),
            'n_classes': config.train['n_classes']
        },
        config=run_config
    )

