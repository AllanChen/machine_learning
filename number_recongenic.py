#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据
mnist =  input_data.read_data_sets("MNIST",one_hot=True)

#每个批次的大小
batch_size = 100

#一共有多少个批次
batch_count = mnist.train.num_examples // batch_size

#定义两个Placeholder
x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])

#创建一个神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W) + b)

#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))

#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

#将结果放在一个布尔值上
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(batch_count):
            batch_xs, batch_ys = mnist.train.next_batch(batch_count)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter " + str(epoch) + "Testing Accuracy"+ str(acc))