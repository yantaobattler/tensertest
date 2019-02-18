import tensorflow as tf


def test1():  # 用Tensorflow计算a=(b+c)∗(c+2)
    sess = tf.Session()

    const = tf.constant(2.)
    b = tf.Variable(2.)
    c = tf.Variable(3.)
    d = b + c
    e = c + const
    a = d * e
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(a))



if __name__ == '__main__':
    test1()
