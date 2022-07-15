import tensorflow.compat.v1 as tf
import random

tf.disable_v2_behavior()
# 产生随机数种子
random.seed()

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
wn = tf.nn.softmax(w)

n1 = wn * x
n2 = tf.reduce_sum(n1)
y = tf.nn.sigmoid(n2)

loss = tf.abs(yTrain -y)
optimizer = tf.train.RMSPropOptimizer(0.1)

train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for i in range(5):
    xData = [int(random.random() * 8 + 93), int(random.random() * 8 + 93), int(random.random() * 8 + 93)]
    xAll = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
    if xAll >= 95:
        yTrainData = 1
    else:
        yTrainData = 0
    result = sess.run([train, x, yTrain, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
    print("大概率三好学生：",result)

    xData = [int(random.random() * 41 + 60), int(random.random() * 41 + 60), int(random.random() * 41 + 60)]
    xAll = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
    if xAll >= 95:
        yTrainData = 1
    else:
        yTrainData = 0
    result = sess.run([train, x, yTrain, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
    print("小概率：",result)
