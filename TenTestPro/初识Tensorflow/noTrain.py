import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x1 = tf.placeholder(dtype=tf.float32)
x2 = tf.placeholder(dtype=tf.float32)
x3 = tf.placeholder(dtype=tf.float32)

# 目标值
yTrain = tf.placeholder(dtype=tf.float32)

w1 = tf.Variable(0.1, dtype=tf.float32)
w2 = tf.Variable(0.1, dtype=tf.float32)
w3 = tf.Variable(0.1, dtype=tf.float32)

n1 = x1 * w1
n2 = x2 * w2
n3 = x3 * w3

y = n1 + n2 + n3

# 误差
loss = tf.abs(y - yTrain)
optimizer = tf.train.RMSPropOptimizer(0.001)
train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# result = sess.run([x1, x2, x3, w1, w2, w3, y], feed_dict={x1: 90, x2: 80, x3: 70})
result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss], feed_dict={x1: 90, x2: 80, x3: 70, yTrain: 85})
print(result)

result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss], feed_dict={x1: 98, x2: 95, x3: 87, yTrain: 96})
print(result)

'''
输出结果如下：
[None, array(90., dtype=float32), array(80., dtype=float32), array(70., dtype=float32), 0.10316052, 0.10316006, 0.103159375, 24.0, array(85., dtype=float32), 61.0]
[None, array(98., dtype=float32), array(95., dtype=float32), array(87., dtype=float32), 0.10554425, 0.10563005, 0.1056722, 28.884804, array(96., dtype=float32), 67.1152]
第一行的61表示loss的数值 也就是85-24的绝对值  
3个w也发生了变化 这是因为调用tf.train.RMSPropOptimizer优化器对参数进行调整  
但是训练次数不够 导致误差较大
下一个文件train.py文件将进行训练
'''