import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import  pandas as pd

filrData = pd.read_csv('upData.txt', dtype=np.float32, header=None)
wholeData = filrData.values.tolist()
rowCount = int(wholeData.size / wholeData[0].size)

goodCount = 0

for i in range(rowCount):
    if wholeData[i][0] * 0.6 + wholeData[i][1] * 0.3 + wholeData[i][2] * 0.1 >= 95:
        goodCount = goodCount + 1

print("wholeData=%s" % wholeData)
print("rowCount=%s" % rowCount)
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
        result = sess.run([train, x, yTrain, wn, b, n2, y, loss], feed_dict={x: wholeData[j][0:3], yTrain: wholeData[j][3]})
        print(result)
