# -*- coding: UTF-8 -*-

"""
Created on 18-1-14

@summary: 测试

@author: dreamhome
"""
import pandas as pd
import numpy as np
import tensorflow as tf

s1 = pd.Series(np.array([4, 3, 2, 1]))
s2 = pd.Series(np.array([5, 6, 7, 8]))
df = pd.DataFrame([s1, s2])
x = df.values
y = x.take(1, 1)

with tf.Session() as sess:
    print(sess.run(tf.random_uniform(
        [2, 2], -1, 1, dtype=tf.float32)))
