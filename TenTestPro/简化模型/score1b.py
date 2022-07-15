import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(shape=[3], dtype=tf.float32)
yTrain = tf.placeholder(shape=[], dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
# softmax函数可以把一个向量规范化后得到一个新的向量，这个新向量中的所有数值相加起来保证为1
wn = tf.nn.softmax(w)

n = x * wn

y = tf.reduce_sum(n)

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.1)

train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for i in range(5):
    result = sess.run([train, x, w, wn, y, yTrain, loss], feed_dict={x: [90, 80, 70], yTrain: 85})
    # 只输出第4项即wn
    print(result[3])
