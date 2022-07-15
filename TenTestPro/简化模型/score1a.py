import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 改成了一个三维的向量存入x
x = tf.placeholder(shape=[3], dtype=tf.float32)
yTrain = tf.placeholder(shape=[], dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)

n = x * w

y = tf.reduce_sum(n)

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.001)

train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

result = sess.run([train, x, w, y, yTrain, loss], feed_dict={x: [90, 80, 70], yTrain: 85})
print(result)

result = sess.run([train, x, w, y, yTrain, loss], feed_dict={x: [98, 95, 87], yTrain: 96})
print(result)