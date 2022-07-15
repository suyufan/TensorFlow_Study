# 进阶： 批量生成随机训练数据
import tensorflow.compat.v1 as tf
import random
import numpy as np

tf.disable_v2_behavior()

random.seed()
rowCount = 5

xData = np.full(shape=(rowCount, 3), fill_value=0, dtype=np.float32)
yTrainData = np.full(shape=(rowCount, 3), fill_value=0, dtype=np.float32)

goodCount = 0

# 生成随机训练数据的循环
for i in range(rowCount):
    xData[i][0] = int(random.random() * 11 + 90)
    xData[i][1] = int(random.random() * 11 + 90)
    xData[i][2] = int(random.random() * 11 + 90)
    xAll = xData[i][0] * 0.6 + xData[i][1] * 0.3 + xData[i][2] * 0.1
    if xAll >= 95:
        yTrainData[i] = 1
        goodCount = goodCount + 1
    else:
        yTrainData[i] = 0

print("xData=%s" % xData)
print("yTrainData=%s" % yTrainData)
print("goodCount=%s" % goodCount)

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
b = tf.Variable(80, dtype=tf.float32)
wn = tf.nn.softmax(w)

n1 = wn * x
n2 = tf.reduce_sum(n1) - b
y = tf.nn.sigmoid(n2)

loss = tf.abs(yTrain -y)
optimizer = tf.train.RMSPropOptimizer(0.1)

train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for i in range(2):
    for j in range(rowCount):
        result = sess.run([train, x, yTrain, wn, b, n2, y, loss], feed_dict={x: xData[j], yTrain: yTrainData[j]})
        print(result)

