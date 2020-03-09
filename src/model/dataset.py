#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse data and generate input_fn for tf.estimators"""

import os
import sys
import abc
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from inspect import getsourcefile
CURRENT_PATH = os.path.abspath(getsourcefile(lambda: 0))
CURRENT_DIR = os.path.dirname(CURRENT_PATH)
PARENT_DIR = CURRENT_DIR[:CURRENT_DIR.rfind(os.path.sep)]
PARENTPRE_DIR = CURRENT_DIR[:PARENT_DIR.rfind(os.path.sep)]
sys.path.insert(0, PARENTPRE_DIR)
from src.model.read_conf import Config


class _CTRDataset(object):
    """Interface for dataset using abstract class"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_file):
        # check file exsits, turn to list so that data_file can be both file or directory.
        assert tf.gfile.Exists(data_file), (
            'data file: {} not found. Please check input data path'.format(data_file))
        if tf.gfile.IsDirectory(data_file):
            data_file_list = [f for f in tf.gfile.ListDirectory(data_file) if not f.startswith('.')]
            data_file = [data_file + '/' + file_name for file_name in data_file_list]
        self._data_file = data_file
        self._conf = Config()
        self._train_conf = self._conf.train
        self._dist_conf = self._conf.distribution
        self._cnn_conf = self._conf.model
        self._shuffle_buffer_size = self._train_conf["num_examples"]
        self._num_parallel_calls = self._train_conf["num_parallel_calls"]
        self._train_epochs = self._train_conf["train_epochs"]

    @abc.abstractmethod
    def input_fn(self, mode, batch_size):
        """
        Abstract input function for train or evaluation (with label),
        abstract method must be implemented in subclasses when instantiate.
        Args:
            mode: `train`, `eval` or `pred`
                train for train mode, do shuffle, repeat num_epochs
                eval for eval mode, no shuffle, no repeat
                pred for pred input_fn, no shuffle, no repeat and no label
            batch_size: Int
        Returns:
            (features, label)
            `features` is a dictionary in which each value is a batch of values for
            that feature; `labels` is a batch of labels.
        """
        raise NotImplementedError('Calling an abstract method.')


class _CsvDataset(_CTRDataset):
    """A class to parse csv data and build input_fn for tf.estimators"""

    def __init__(self, data_file):
        super(_CsvDataset, self).__init__(data_file)
        self._pos_sample_loss_weight = self._train_conf["pos_sample_loss_weight"]
        self._neg_sample_loss_weight = self._train_conf["neg_sample_loss_weight"]
        self._use_weight = False
        if self._pos_sample_loss_weight is not None and self._neg_sample_loss_weight is not None:
            self._use_weight = True
            print("Using weight column to use weighted loss.")
        self._multivalue = self._train_conf["multivalue"]
        self._is_distribution = self._dist_conf["is_distribution"]
        cluster = self._dist_conf["cluster"]
        job_name = self._dist_conf["job_name"]
        task_index = self._dist_conf["task_index"]
        self._num_workers = 1 + len(cluster["worker"])  # must have 1 chief worker
        self._worker_index = task_index if job_name == "worker" else self._num_workers-1
        self._feature = self._conf.get_feature_name()  # all features
        self._feature_used = self._conf.get_feature_name('used')  # used features
        self._feature_unused = self._conf.get_feature_name('unused')  # unused features
        self._feature_conf = self._conf.read_feature_conf()  # feature conf dict
        self._csv_defaults = self._column_to_csv_defaults()

    def _column_to_csv_defaults(self):
        """parse columns to record_defaults param in tf.decode_csv func
        Return:
            OrderedDict {'feature name': [''],...}
        """
        csv_defaults = OrderedDict()
        csv_defaults['label'] = [0]  # first label default, empty if the field is must
        for f in self._feature:
            if f in self._feature_conf:  # used features
                conf = self._feature_conf[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':  # identity category column need int type
                        csv_defaults[f] = [0]
                    else:
                        csv_defaults[f] = [b'']
                elif conf['type'] == 'continuous':
                    csv_defaults[f] = [0.0]
                elif conf['type'] == 'no_featurecolumn':
                    csv_defaults[f] = [b'0']
                else:
                    csv_defaults[f] = [0.0]  # 0.0 for float32
            else:  # unused features
                csv_defaults[f] = [b'']
        return csv_defaults

    def _parse_csv(self, is_pred=False, field_delim='\t', na_value='-', multivalue_delim=','):
        """Parse function for csv data
        Args:
            is_pred: bool, defaults to False
                True for pred mode, parse input data with label
                False for train or eval mode, parse input data without label
            field_delim: csv fields delimiter, defaults to `\t`
            na_value: use csv defaults to fill na_value
            multivalue: bool, defaults to False
                True for csv data with multivalue features.
                eg:   f1       f2   ...
                    a, b, c    1    ...
                     a, c      2    ...
                     b, c      0    ...
            multivalue_delim: multivalue feature delimiter, defaults to `,`
        Returns:
            feature dict: {feature: Tensor ... }
        """
        # if is_pred:
        #     self._csv_defaults.pop('label')
        csv_defaults = self._csv_defaults
        multivalue = self._multivalue

        def parser(value):
            """Parse train and eval data with label
            Args:
                value: Tensor("arg0:0", shape=(), dtype=string)
            """
            # `tf.decode_csv` return rank 0 Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>
            # na_value fill with record_defaults
            columns = tf.decode_csv(
                value, record_defaults=list(csv_defaults.values()),
                field_delim=field_delim, use_quote_delim=False, na_value=na_value)
            features = dict(zip(csv_defaults.keys(), columns))
            for f, tensor in list(features.items()):
                if f in self._feature_unused:
                    features.pop(f)  # remove unused features
                    continue
                if multivalue:  # split tensor
                    if isinstance(csv_defaults[f][0], bytes) and f != 'user_id':
                        # input must be rank 1, return SparseTensor
                        # print(st.values)  # <tf.Tensor 'StringSplit_11:1' shape=(?,) dtype=string>
                        features[f] = tf.string_split([tensor], multivalue_delim).values  # tensor shape (?,)
                    else:
                        features[f] = tf.expand_dims(tensor, 0)  # change shape from () to (1,)
                if f == 'expose_goodsid':
                    expose_goodsid = tf.random_shuffle(features[f])
                    features[f] = expose_goodsid[0:10]
                if f == 'sample_goodsid':
                    sample_goodsid = tf.random_shuffle(features[f])
                    features[f] = sample_goodsid[0:20]
                if f == 'goodsid_clicked':
                    features[f] = features[f][0:20]
                if f == 'goodsid_carted':
                    features[f] = features[f][0:20]
                if f == 'goodsid_collected':
                    features[f] = features[f][0:20]
                if f == 'goodsid_buyed':
                    features[f] = features[f][0:20]
            if is_pred:
                return features
            else:
                label = features.pop('label')
                ctr = tf.cond(tf.less_equal(1, label[0]), lambda:tf.constant([1.0]), lambda:tf.constant([0.0]))
                # if use_weight:
                #     pred = labels[0] if multivalue else labels  # pred must be rank 0 scalar
                #     pos_weight, neg_weight = pos_w or 1, neg_w or 1
                #     weight = tf.cond(pred, lambda: pos_weight, lambda: neg_weight)
                #     features["weight_column"] = [weight]  # padded_batch need rank 1
                return features, {'ctr_label': ctr}
        return parser

    def input_fn(self, mode, batch_size):
        assert mode in {'train', 'eval', 'pred'}, (
            'mode must in `train`, `eval`, or `pred`, found {}'.format(mode))
        tf.logging.info('Parsing input csv files: {}'.format(self._data_file))
        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(self._data_file)
        # Use `Dataset.map()` to build a pair of a feature dictionary
        # and a label tensor for each example.
        # Shuffle, repeat, and batch the examples.
        dataset = dataset.map(
            self._parse_csv(is_pred=(mode == 'pred'), field_delim='\t'),
            num_parallel_calls=self._num_parallel_calls)
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size, seed=123)
            dataset = dataset.repeat(self._train_epochs)  # define outside loop

        dataset = dataset.prefetch(2 * batch_size)
        if self._multivalue:
            padding_dic = {k: [None] for k in self._feature_used}
            if mode == 'pred':
                padding_dic.update({'label': [None]})
                padded_shapes = padding_dic
            else:
                padded_shapes = (padding_dic, {'ctr_label': [None]})
            dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        else:
            # batch(): each element tensor must have exactly same shape, change rank 0 to rank 1
            dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()


def input_fn(csv_data_file, mode, batch_size):
    """Combine input_fn for tf.estimators
    Combine both csv and image data; combine both train and pred mode.
    set img_data_file None to use only csv data
    """
    if mode == 'pred':
        features = _CsvDataset(csv_data_file).input_fn(mode, batch_size)
        return features

    else:
        features, label = _CsvDataset(csv_data_file).input_fn(mode, batch_size)
        return features, label


def _input_tensor_test(data_file, batch_size=2):
    """test for categorical_column and cross_column input."""
    sess = tf.InteractiveSession()
    features, labels = _CsvDataset(data_file).input_fn('train', batch_size=batch_size)
    features_print = sess.run(features)
    print(features_print)
    # categorical_column* can handle multivalue feature as a multihot
    from src.model.build_columns import _build_model_columns
    CONFIG = Config()
    feature_columns = _build_model_columns(CONFIG)
    user_emb = tf.feature_column.input_layer(features, feature_columns)

    sess.run(tf.global_variables_initializer())  # fix Attempting to use uninitialized value error.
    sess.run(tf.tables_initializer())  # fix Table not initialized error.
    a = sess.run(user_emb)
    print(a)



if __name__ == '__main__':
    from os.path import dirname, abspath

    csv_path = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'data/train/train_data')
    # _input_tensor_test(csv_path)
    sess = tf.InteractiveSession()

    data = input_fn(csv_path, 'train', 2)
    for i in range(10):
        a = sess.run(data)
        print(a)
        print('\n')
