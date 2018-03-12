# -*- coding: UTF-8 -*-

"""
Created on 18-1-25

@summary: 构建计算图

@author: dreamhome
"""
from model import *


# 构建计算图
tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    # 获取输入占位符
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = get_inputs()
    # 获取User的4个嵌入向量
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(
        uid, user_gender, user_age, user_job)
    # 得到用户特征
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
    # 获取电影ID的嵌入向量
    movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
    # 获取电影类型的嵌入向量
    movie_categories_embed_layer = get_movie_categories_layers(
        movie_categories)
    # 获取电影名的特征向量
    pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles)
    # 得到电影特征
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(
        movie_id_embed_layer, movie_categories_embed_layer, dropout_layer)
    # 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
    with tf.name_scope("inference"):
        # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
        #         inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  #(?, 200)
        #         inference = tf.layers.dense(inference_layer, 1,
        #                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        #                                     kernel_regularizer=tf.nn.l2_loss, name="inference")
        # 简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
        inference = tf.matmul(
            user_combine_layer_flat,
            tf.transpose(movie_combine_layer_flat))

    with tf.name_scope("loss"):
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, inference)
        loss = tf.reduce_mean(cost)
    # 优化损失
    # train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)  # cost
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)


