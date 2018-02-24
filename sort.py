import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x_data = np.linspace(-0.5,0.5, 500)[:,np.newaxis]
noise =  np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise


x = tf.placeholder(tf.float32,[None, 1])
y = tf.placeholder(tf.float32,[None, 1])


Weights_L1 = tf.Variable(tf.random_normal([1,10]))
baises_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + baises_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)


Weights_L2 = tf.Variable(tf.random_normal([10,1]))
baises_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + baises_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)


loss = tf.reduce_mean(tf.square(y-prediction))


train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()
    # pylab.show()
