#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@Time    : 2020/3/5 20:03
@Author  : zz
@File    : train.py
@Software: PyCharm
@Desc    : 整体的调度文件，增加性能检查工具的使用
"""
import os
import sys
import shutil

import tensorflow as tf

from read_conf import Config
from build_columns import _build_model_columns
from build_model import _build_model_function
from dataset import input_fn

CONFIG = Config()
CONFIG_TRAIN = CONFIG.train
CONFIG_RUN = CONFIG.runconfig
CONFIG_MODEL = CONFIG.model

configuration_params = {
    'max_steps_without_increase': 1000,
    'min_steps': 100000,
    'run_every_steps': 100
}


def train_and_eval(model):
    """
    :param model: 声明的estimator实例
    :return: None
    :usage: 进行模型训练，并在指定步长的时候进行结果评估
    """
    timeline_hook = tf.train.ProfilerHook(save_steps=100, output_dir=os.path.join(
        os.getcwd(), './timeline_track'
    ))

    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        model,
        metric_name='auc',
        max_steps_without_increase=configuration_params['max_steps_without_increase'],
        # maximum number of training steps with no decrease in the given metric.
        min_steps=configuration_params['min_steps'],  # stop is never requested if global step is less than this value
        run_every_steps=configuration_params['run_every_steps'],
        run_every_secs=None
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(os.path.join(os.getcwd(),
                                               CONFIG_TRAIN['train_data']),
                                  'train', CONFIG_TRAIN['batch_size']),
        hooks=[hook, timeline_hook]
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(os.path.join(os.getcwd(),
                                               CONFIG_TRAIN['test_data']),
                                  'eval', 128),
        steps=CONFIG.evalconfig['steps'],
        throttle_secs=30
        )
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def main(argv=[]):
    cur_dir = os.getcwd()
    model_dir = os.path.join(cur_dir, CONFIG_TRAIN['model_dir'],
                             CONFIG_TRAIN['model_type'])

    print("the model_dir is: {}".format(model_dir))

    # 全量训练的时候将之前的模型进行删除
    if not CONFIG_TRAIN['keep_train']:
        shutil.rmtree(model_dir, ignore_errors=True)
        print('Remove model directory: {}'.format(model_dir))

    # 构建feature_columns
    user_feature_columns, goods_feature_columns = _build_model_columns(CONFIG)

    model = _build_model_function(model_dir, config=CONFIG, feature_columns={'user_columns': user_feature_columns,
                                                                             'good_columns': goods_feature_columns})

    tf.logging.info('Build estimator: {}'.format(model))

    train_and_eval(model)

    print(model_dir)
    return 0


if __name__ == "__main__":
    # Set to INFO for tracking training, default is WARN, ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]])
