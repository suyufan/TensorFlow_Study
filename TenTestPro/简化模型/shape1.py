import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(dtype=tf.float32)

xShape = tf.shape(x)

sess = tf.Session()

result = sess.run(xShape, feed_dict={x: 8})
print(result)

result = sess.run(xShape, feed_dict={x: [1, 2, 3]})
print(result)

result = sess.run(xShape, feed_dict={x: [[1, 2, 3], [3, 6, 9]]})
print(result)