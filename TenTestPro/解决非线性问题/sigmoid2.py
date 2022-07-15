import tensorflow.compat.v1 as tf
import random

tf.disable_v2_behavior()

random.seed()

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
b = tf.Variable(80, dtype=tf.float32)
wn = tf.nn.softmax(w)

n1 = wn * x
# 加上可变参数b 否则y的值要么是0 要么是1  因为他的绝大多数取值都是0/1 只有在【-5，5】之间变化剧烈
n2 = tf.reduce_sum(n1) - b
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
    print("大：",result)

    xData = [int(random.random() * 41 + 60), int(random.random() * 41 + 60), int(random.random() * 41 + 60)]
    xAll = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
    if xAll >= 95:
        yTrainData = 1
    else:
        yTrainData = 0
    result = sess.run([train, x, yTrain, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
    print("小概率-------------：",result)
