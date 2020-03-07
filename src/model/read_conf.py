#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read All Configuration from ./conf/*.yaml
1.检查字段类型
2.返回配置
"""
import os
import yaml

from os.path import dirname, abspath

# BASE_DIR = os.path.join(DIR, 'conf')
BASE_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'conf')
# 特征每列对应的名字
SCHEMA_CONF_FILE = 'schema.yaml'
# 特征配置文件，包括特征处理方式和分类
FEATURE_CONF_FILE = 'feature.yaml'
# 模型配置文件
MODEL_CONF_FILE = 'model.yaml'
# 训练过程中的相关参数
TRAIN_CONF_FILE = 'train.yaml'


class Config(object):
    """Config class
    Class attributes: config, train, distribution, model, runconfig, serving
    """
    def __init__(self,
                 schema_conf_file=SCHEMA_CONF_FILE,
                 feature_conf_file=FEATURE_CONF_FILE,
                 model_conf_file=MODEL_CONF_FILE,
                 train_conf_file=TRAIN_CONF_FILE):
        self._schema_conf_file = os.path.join(BASE_DIR, schema_conf_file)
        self._feature_conf_file = os.path.join(BASE_DIR, feature_conf_file)
        self._model_conf_file = os.path.join(BASE_DIR, model_conf_file)
        self._train_conf_file = os.path.join(BASE_DIR, train_conf_file)

    @staticmethod
    def _check_feature_conf(feature, valid_feature_name, **kwargs):
        type_ = kwargs["type"]
        trans = kwargs["transform"]
        param = kwargs["parameter"]
        if type_ is None:
            raise ValueError("Type are required in feature conf, "
                             "found empty value for feature `{}`".format(feature))
        if feature not in valid_feature_name:
            raise ValueError("Invalid feature name `{}` in feature conf, "
                             "must be consistent with schema conf".format(feature))
        assert type_ in {'category', 'continuous', 'no_featurecolumn'}, (
            "Invalid type `{}` for feature `{}` in feature conf, "
            "must be 'category' or 'continuous'".format(type_, feature))
        # check transform and parameter
        if type_ == 'category':
            assert trans in {'hash_bucket', 'identity', 'vocab'}, (
                "Invalid transform `{}` for feature `{}` in feature conf, "
                "must be one of `hash_bucket`, `vocab`, `identity`.".format(trans, feature))
            if trans == 'hash_bucket' or trans == 'identity':
                if not isinstance(param, int):
                    raise TypeError('Invalid parameter `{}` for feature `{}` in feature conf, '
                                    '{} parameter must be an integer.'.format(param, feature, trans))
            elif trans == 'vocab':
                if not isinstance(param, (tuple, list)):
                    raise TypeError('Invalid parameter `{}` for feature `{}` in feature conf, '
                                    'vocab parameter must be a list.'.format(param, feature))
        elif type_ == 'continuous':
            normalization, boundaries = param['normalization'], param['boundaries']
            if trans:
                assert trans in {'min_max', 'log', 'standard'}, \
                    "Invalid transform `{}` for feature `{}` in feature conf, " \
                    "continuous feature transform must be `min_max` or `log` or `standard`.".format(trans, feature)
                if trans == 'min_max' or 'standard':
                    if not isinstance(normalization, (list, tuple)) or len(normalization) != 2:
                        raise TypeError('Invalid normalization parameter `{}` for feature `{}` in feature conf, '
                                        'must be 2 elements list for `min_max` or `standard` scaler.'.format(normalization, feature))
                if trans == 'min_max':
                    min_, max_ = normalization
                    if not isinstance(min_, (float, int)) or not isinstance(max_, (float, int)):
                        raise TypeError('Invalid normalization parameter `{}` for feature `{}` in feature conf, '
                                        'list elements must be int or float.'.format(normalization, feature))
                    assert min_ < max_, ('Invalid normalization parameter `{}` for feature `{}` in feature conf, '
                                         '[min, max] list elements must be min<max'.format(normalization, feature))
                elif trans == 'standard':
                    mean, std = normalization
                    if not isinstance(mean, (float, int)):
                        raise TypeError('Invalid normalization parameter `{}` for feature `{}` in feature conf, '
                                        'parameter mean must be int or float.'.format(mean, feature))
                    if not isinstance(std, (float, int)) or std <= 0:
                            raise TypeError('Invalid normalization parameter `{}` for feature `{}` in feature conf, '
                                            'parameter std must be a positive number.'.format(std, feature))
            if boundaries:
                if not isinstance(boundaries, (tuple, list)):
                    raise TypeError('Invalid parameter `{}` for feature `{}` in feature conf, '
                                    'discretize parameter must be a list.'.format(boundaries, feature))
                else:
                    for v in boundaries:
                        assert isinstance(v, (int, float)), \
                            "Invalid parameter `{}` for feature `{}` in feature conf, " \
                            "discretize parameter element must be integer or float.".format(boundaries, feature)
        else:
            assert type_ == 'no_featurecolumn', (
                "Invalid transform `{}` for feature `{}` in feature conf, "
                "must be one of `hash_bucket`, `vocab`, `identity`.".format(trans, feature))

    @staticmethod
    def _check_cross_feature_conf(features, feature_conf, **kwargs):
        features_list = [f.strip() for f in features.split('&')]
        hash_bucket_size = kwargs["hash_bucket_size"]
        is_deep = kwargs["is_deep"]
        assert len(features_list) > 1, (
            'Invalid cross feature name `{}` in cross feature conf,'
            'at least 2 features'.format(features))
        for f in features_list:
            if f not in feature_conf:
                raise ValueError("Invalid cross feature name `{}` in cross feature conf, "
                                 "must be consistent with feature conf".format(features))
            if feature_conf[f]['type'] == 'continuous':
                assert feature_conf[f]['parameter']['boundaries'] is not None, \
                    'Continuous feature must be set bounaries to be bucketized in feature conf as cross feature'
        if hash_bucket_size:
            assert isinstance(hash_bucket_size, (int, float)), (
                'Invalid hash_bucket_size `{}` for features `{}` in cross feature conf, ' 
                'expected int or float'.format(hash_bucket_size, features))
        if is_deep:
            assert is_deep in {0, 1}, (
                'Invalid is_deep `{}` for features `{}`, ' 
                'expected 0 or 1.'.format(is_deep, features))

    @staticmethod
    def _check_numeric(key, value):
        if not isinstance(value, (int, float)):
            raise ValueError('Numeric type is required for key `{}`, found `{}`.'.format(key, value))

    @staticmethod
    def _check_string(key, value):
        if not isinstance(value, str):
            raise ValueError('String type is required for key `{}`, found `{}`.'.format(key, value))

    @staticmethod
    def _check_bool(key, value):
        if value not in {True, False, 1, 0}:
            raise ValueError('Bool type is required for key `{}`, found `{}`.'.format(key, value))

    @staticmethod
    def _check_list(key, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError('List type is required for key `{}`, found `{}`.'.format(key, value))

    @staticmethod
    def _check_required(key, value):
        if value is None:
            raise ValueError('Required type for key `{}`, found None.'.format(key))


    @property
    def config(self):
        return self._read_train_conf()

    @property
    def train(self):
        return self._read_train_conf()["train"]

    @property
    def evalconfig(self):
        return self._read_train_conf()["evalconfig"]

    @property
    def featureconfig(self):
        return self._read_train_conf()["featureconfig"]

    @property
    def distribution(self):
        return self._read_train_conf()["distribution"]

    @property
    def runconfig(self):
        return self._read_train_conf()["runconfig"]

    @property
    def model(self):
        return self._read_model_conf()

    def read_schema(self):
        with open(self._schema_conf_file) as f:
            return {k: v.lower() for k, v in yaml.safe_load(f).items()}

    # feature.yaml文件
    def read_feature_conf(self):
        with open(self._feature_conf_file) as f:
            feature_conf = yaml.safe_load(f)
            valid_feature_name = self.read_schema().values()
            for feature, conf in feature_conf.items():
                self._check_feature_conf(feature.lower(), valid_feature_name, **conf)
            return feature_conf

    def _read_model_conf(self):
        # required string params
        req_str_keys = ['linear_optimizer', 'dnn_optimizer', 'dnn_connected_mode', 'dnn_activation_function']
        # optional int or float params
        opt_num_keys = ['linear_initial_learning_rate', 'linear_decay_rate', 'dnn_initial_learning_rate',
                        'dnn_decay_rate', 'dnn_l1', 'dnn_l2']
        # optional bool params
        opt_bool_keys = ['dnn_batch_normalization']
        #
        req_list_keys = ['dnn_hidden_units']
        with open(self._model_conf_file) as f:
            model_conf = yaml.safe_load(f)
            for k, v in model_conf.items():
                if k in req_str_keys:
                    self._check_required(k, v)
                    self._check_string(k, v)
                elif k in opt_num_keys:
                    if v:
                        self._check_numeric(k, v)
                elif k in opt_bool_keys:
                    if v:
                        self._check_bool(k, v)
                elif k in req_list_keys:
                    self._check_required(k, v)
                    self._check_list(k, v)
            return model_conf

    def _read_train_conf(self):
        req_str_keys = ['model_dir', 'model_type', 'train_data', 'test_data']
        req_num_keys = ['train_epochs', 'epochs_per_eval', 'batch_size', 'num_examples']
        opt_num_keys = ['pos_sample_loss_weight', 'neg_sample_loss_weight', 'num_parallel_calls']
        req_bool_key = ['keep_train', 'multivalue', 'dynamic_train']
        with open(self._train_conf_file) as f:
            train_conf = yaml.safe_load(f)
            for k, v in train_conf['train'].items():
                if k in req_str_keys:
                    self._check_required(k, v)
                    self._check_string(k, v)
                elif k in req_num_keys:
                    self._check_required(k, v)
                    self._check_numeric(k, v)
                elif k in opt_num_keys:
                    if v:
                        self._check_numeric(k, v)
                elif k in req_bool_key:
                    self._check_required(k, v)
                    self._check_bool(k, v)
            return train_conf

    def get_feature_name(self, feature_type='all'):
        """
        Args:
         feature_type: one of {'all', 'used', 'category', 'continuous'}
        Return: feature name list
        """
        feature_conf_dic = self.read_feature_conf()
        feature_list = list(self.read_schema().values())
        feature_list.remove('if_clicked')
        if feature_type == 'all':
            return feature_list
        elif feature_type == 'used':
            return feature_conf_dic.keys()
        elif feature_type == 'unused':
            return set(feature_list) - set(feature_conf_dic.keys())
        elif feature_type == 'category':
            return [feature for feature, conf in feature_conf_dic.items() if conf['type'] == 'category']
        elif feature_type == 'continuous':
            return [feature for feature, conf in feature_conf_dic.items() if conf['type'] == 'continuous']
        else:
            raise ValueError("Invalid parameter, must be one of 'all', 'used', 'category, 'continuous")


def _test():
    config = Config()
    """test for Config methods"""
    print('\nTrain config:')
    print(config.config)
    print(config.train)
    print(config.runconfig)
    print(config.train["model_dir"])

    print('\nModel conf:')
    for k, v in config.model.items():
        print(k, v)

    feature_conf_dic = config.read_feature_conf()
    print('\nFeature conf:')
    for k, v in feature_conf_dic.items():
        print(k, v)

    category_feature = config.get_feature_name('category')
    print('\nCategory feature:')
    print(category_feature)

    members = [m for m in Config.__dict__ if not m.startswith('_')]
    print('\nConfig class members:')
    print(members)

    print(config.evalconfig)


if __name__ == '__main__':
    _test()




