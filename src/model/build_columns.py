#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@Time    : 2020/3/5 21:04
@Author  : zz
@File    : build_columns.py
@Software: PyCharm
@Desc    : 构建双塔模型对应的feature_columns，主要要输出两部分的cols【用户侧】+【商品侧】
"""
import numpy as np
import tensorflow as tf
from read_conf import Config


def _build_model_columns(CONF):
    """
    Build wide and deep feature columns from custom feature conf using tf.feature_column API
    wide_columns: category features + cross_features + [discretized continuous features]
    columns: continuous features + category features(onehot or embedding for sparse features) + [cross_features(embedding)]
    Return:
        _CategoricalColumn and __DenseColumn instance in tf.feature_column API
    """
    def embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim**0.25))))

    def normalizer_fn_builder(scaler, normalization_params):
        """normalizer_fn builder"""
        if scaler == 'min_max':
            return lambda x: (x-normalization_params[0]) / (normalization_params[1]-normalization_params[0])
        elif scaler == 'standard':
            return lambda x: (x-normalization_params[0]) / normalization_params[1]
        else:
            return lambda x: tf.log(x)

    feature_conf_dic = CONF.read_feature_conf()

    tf.logging.info('Total used feature class: {}'.format(len(feature_conf_dic)))
    # tf.logging.info('Total used cross feature class: {}'.format(len(cross_feature_list)))

    user_columns = []
    good_columns = []
    user_cols_dim = 0
    good_cols_dim = 0
    # generate feature columns for DSSM
    for feature, conf in feature_conf_dic.items():
        f_type, f_tran, f_param, f_flag = conf['type'], conf['transform'], conf['parameter'], conf['flag']
        if f_type == 'category':
            if f_tran == 'hash_bucket':
                hash_bucket_size = f_param
                embed_dim = embedding_dim(hash_bucket_size)
                col = tf.feature_column.categorical_column_with_hash_bucket(feature,
                                                                            hash_bucket_size=hash_bucket_size,
                                                                            dtype=tf.string)
                emb_col = tf.feature_column.embedding_column(col,
                                    dimension=embed_dim,
                                    combiner='mean',
                                    initializer=None,
                                    ckpt_to_load_from=None,
                                    tensor_name_in_ckpt=None,
                                    max_norm=None,
                                    trainable=True)
                if f_flag == 'user':
                    user_columns.append(emb_col)
                    user_cols_dim += embed_dim
                else:
                    good_columns.append(emb_col)
                    good_cols_dim += embed_dim
            elif f_tran == 'vocab':
                embed_dim = embedding_dim(len(f_param))
                col = tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                vocabulary_list=list(
                                                                                    map(str, f_param)
                                                                                ),
                                                                                dtype=None,
                                                                                default_value=-1,
                                                                                num_oov_buckets=0)
                embedding_col = tf.feature_column.embedding_column(col,
                            dimension=embed_dim,
                            combiner='mean',
                            initializer=None,
                            ckpt_to_load_from=None,
                            tensor_name_in_ckpt=None,
                            max_norm=None,
                            trainable=True) 
                if f_flag == 'user':
                    user_columns.append(embedding_col)
                    user_cols_dim += embed_dim
                else:
                    good_columns.append(embedding_col)
                    good_cols_dim += embed_dim
            elif f_tran == 'identity':
                num_buckets = f_param
                # Values outside range will result in default_value if specified, otherwise it will fail.
                col = tf.feature_column.categorical_column_with_identity(feature,
                                                                         num_buckets=num_buckets,
                                                                         default_value=0)
                if f_flag == 'user':
                    user_columns.append(tf.feature_column.indicator_column(col))
                    user_cols_dim += num_buckets
                else:
                    good_columns.append(tf.feature_column.indicator_column(col))
                    good_cols_dim += num_buckets
        elif f_type == 'continuous':
            normalizaton, boundaries = f_param["normalization"], f_param["boundaries"]
            if f_tran is None:
                normalizer_fn = None
            else:
                normalizer_fn = normalizer_fn_builder(f_tran, tuple(normalizaton))
            col = tf.feature_column.numeric_column(feature,
                                                   shape=(1,),
                                                   default_value=0,
                                                   dtype=tf.float32,
                                                   normalizer_fn=normalizer_fn)
            if boundaries:
                bucketized_feature_column = tf.feature_column.bucketized_column(
                    source_column=col,
                    boundaries=boundaries
                )
                emb_col = tf.feature_column.embedding_column(bucketized_feature_column,
                                                               dimension=embedding_dim(len(boundaries)),
                                                               combiner='mean',
                                                               initializer=None,
                                                               ckpt_to_load_from=None,
                                                               tensor_name_in_ckpt=None,
                                                               max_norm=None,
                                                               trainable=True)
                if f_flag == 'user':
                    user_columns.append(emb_col)
                    user_cols_dim += embedding_dim(len(boundaries))
                else:
                    good_columns.append(emb_col)
                    good_cols_dim += embedding_dim(len(boundaries))
            else:
                if f_flag == 'user':
                    user_columns.append(col)
                    user_cols_dim += 1
                else:
                    good_columns.append(col)
                    good_cols_dim += 1
        else:
            continue

    tf.logging.info("The user's feature columns cnt: {}".format(len(user_columns)))
    tf.logging.info("The good's feature columns cnt: {}".format(len(good_columns)))
    tf.logging.info('Build total {} columns'.format(len(user_columns) + len(good_columns)))
    tf.logging.info('')
    tf.logging.info("the user's feature columns:'")
    for col in user_columns:
        tf.logging.debug('columns: {}'.format(col))
    tf.logging.info('')
    tf.logging.info("the good's feature columns:")
    for col in good_columns:
        tf.logging.debug('columns: {}'.format(col))

    tf.logging.info('user dim is:{}, good dim is:{}, all dimension is: {}'.format(user_cols_dim,
                                                                                  good_cols_dim,
                                                                                  user_cols_dim + good_cols_dim))

    return user_columns, good_columns


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    config = Config()
    user_columns, good_columns = _build_model_columns(config)
