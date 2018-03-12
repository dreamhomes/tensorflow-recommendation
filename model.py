# -*- coding: UTF-8 -*-

"""
Created on 18-1-16

@summary: 建立的推荐模型

@author: dreamhome
"""
import tensorflow as tf
import pickle

# 从本地文件中读取数据

# title_count：Title字段的长度（15）
# title_set：Title文本的集合
# genres2int：电影类型转数字的字典
# features：是输入X
# targets_values：是学习目标y
# ratings：评分数据集的Pandas对象
# users：用户数据集的Pandas对象
# movies：电影数据的Pandas对象
# data：三个数据集组合在一起的Pandas对象
# movies_orig：没有做数据处理的原始电影数据
# users_orig：没有做数据处理的原始用户数据

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
    open('preprocess.p', mode='rb'))

# 参数

# 嵌入矩阵的维度
embed_dim = 32
# 用户ID个数
uid_max = max(features.take(0, 1)) + 1  # 6040
# 性别个数
gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2
# 年龄类别个数
age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7
# 职业个数
job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21

# 电影ID个数
movie_id_max = max(features.take(1, 1)) + 1  # 3952
# 电影类型个数
movie_categories_max = max(genres2int.values()) + 1  # 18 + 1 = 19
# 电影名单词个数
movie_title_max = len(title_set)  # 5216

# 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

# 电影名长度
sentences_size = title_count  # = 15
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
# 文本卷积核数量
filter_num = 8

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

# 超参

# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './save'


def get_inputs():
    """
    神经网络输入
    :return:
    """
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    movie_categories = tf.placeholder(
        tf.int32, [None, 18], name="movie_categories")
    movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.placeholder(tf.float32, name="LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob


def get_user_embedding(uid, user_gender, user_age, user_job):
    """
    用户嵌入矩阵
    :param uid: 用户ID
    :param user_gender: 性别
    :param user_age: 年龄
    :param user_job: 职业
    :return:
    """
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform(
            [uid_max, embed_dim], minval=-1, maxval=1), name="uid_embed_matrix")
        uid_embed_layer = tf.nn.embedding_lookup(
            uid_embed_matrix, uid, name="uid_embed_layer")

        gender_embed_matrix = tf.Variable(tf.random_uniform(
            [gender_max, embed_dim // 2], -1, 1), name="gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(
            gender_embed_matrix, user_gender, name="gender_embed_layer")

        age_embed_matrix = tf.Variable(tf.random_uniform(
            [age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(
            age_embed_matrix, user_age, name="age_embed_layer")

        job_embed_matrix = tf.Variable(tf.random_uniform(
            [job_max, embed_dim // 2], -1, 1), name="job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(
            job_embed_matrix, user_job, name="job_embed_layer")

    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(
        uid_embed_layer,
        gender_embed_layer,
        age_embed_layer,
        job_embed_layer):
    """
    将User的嵌入矩阵一起全连接生成User的特征
    :param uid_embed_layer:
    :param gender_embed_layer:
    :param age_embed_layer:
    :param job_embed_layer:
    :return:
    """
    with tf.name_scope("user_fc"):
        # 第一层全连接
        uid_fc_layer = tf.layers.dense(
            uid_embed_layer,
            embed_dim,
            name="uid_fc_layer",
            activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(
            gender_embed_layer,
            embed_dim,
            name="gender_fc_layer",
            activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(
            age_embed_layer,
            embed_dim,
            name="age_fc_layer",
            activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(
            job_embed_layer,
            embed_dim,
            name="job_fc_layer",
            activation=tf.nn.relu)

        print uid_fc_layer

        # 第二层全连接
        user_combine_layer = tf.concat(
            [uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  # (?, 1, 128)
        user_combine_layer = tf.contrib.layers.fully_connected(
            user_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])

    return user_combine_layer, user_combine_layer_flat


def get_movie_id_embed_layer(movie_id):
    """
    定义Movie ID的嵌入矩阵
    :param movie_id:
    :return:
    """
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform(
            [movie_id_max, embed_dim], -1, 1), name="movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(
            movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
    return movie_id_embed_layer


def get_movie_categories_layers(movie_categories):
    """
    对电影类型的多个嵌入向量做加和
    :param movie_categories:
    :return:
    """
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform(
            [movie_categories_max, embed_dim], -1, 1), name="movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(
            movie_categories_embed_matrix,
            movie_categories,
            name="movie_categories_embed_layer")
        if combiner == "sum":
            movie_categories_embed_layer = tf.reduce_sum(
                movie_categories_embed_layer, axis=1, keep_dims=True)
    #     elif combiner == "mean":

    return movie_categories_embed_layer


def get_movie_cnn_layer(movie_titles):
    """
    Movie Title的文本卷积网络实现
    :param movie_titles:
    :return:
    """
    # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform(
            [movie_title_max, embed_dim], -1, 1), name="movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(
            movie_title_embed_matrix, movie_titles, name="movie_title_embed_layer")
        movie_title_embed_layer_expand = tf.expand_dims(
            movie_title_embed_layer, -1)

    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal(
                [window_size, embed_dim, 1, filter_num], stddev=0.1), name="filter_weights")
            filter_bias = tf.Variable(
                tf.constant(
                    0.1,
                    shape=[filter_num]),
                name="filter_bias")

            conv_layer = tf.nn.conv2d(
                movie_title_embed_layer_expand, filter_weights, [
                    1, 1, 1, 1], padding="VALID", name="conv_layer")
            relu_layer = tf.nn.relu(
                tf.nn.bias_add(
                    conv_layer,
                    filter_bias),
                name="relu_layer")

            maxpool_layer = tf.nn.max_pool(
                relu_layer, [
                    1, sentences_size - window_size + 1, 1, 1], [
                    1, 1, 1, 1], padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # Dropout层
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
        max_num = len(window_sizes) * filter_num
        pool_layer_flat = tf.reshape(
            pool_layer, [-1, 1, max_num], name="pool_layer_flat")

        dropout_layer = tf.nn.dropout(
            pool_layer_flat,
            dropout_keep,
            name="dropout_layer")
    return pool_layer_flat, dropout_layer


def get_movie_feature_layer(
        movie_id_embed_layer,
        movie_categories_embed_layer,
        dropout_layer):
    """
    将Movie的各个层一起做全连接
    :param movie_id_embed_layer:
    :param movie_categories_embed_layer:
    :param dropout_layer:
    :return:
    """
    with tf.name_scope("movie_fc"):
        # 第一层全连接
        movie_id_fc_layer = tf.layers.dense(
            movie_id_embed_layer,
            embed_dim,
            name="movie_id_fc_layer",
            activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(
            movie_categories_embed_layer,
            embed_dim,
            name="movie_categories_fc_layer",
            activation=tf.nn.relu)

        # 第二层全连接
        movie_combine_layer = tf.concat(
            [movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(
            movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer, movie_combine_layer_flat
