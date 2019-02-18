import tensorflow as tf
import random
# simple print
# node1 = tf.constant(2)
# node2 = tf.constant(3)
# sum_node = node1 * node2
# print(node1)
# print(node2)
# print(sum_node)
# print('*' * 10)
#
# session = tf.Session()
# print(session.run([node1, node2, sum_node]))  run里面可以是list


# 带输入
# input_node = tf.placeholder(tf.int32)  # 这是一个节点，会有一个数字的输入，用placeholder占位
# node3 = tf.constant(3)
# sum_node = node3 * input_node
# seession = tf.Session()
# print(seession.run(sum_node, feed_dict={input_node: 4}))  # 带placeholder的，需要在run的时候加一个字典feed_dict

# 带变量初始化方法1
# 要创建变量，请使用 tf.get_variable ()。
# tf.get_variable () 的前两个参数是必需的，其余是可选的。它们是 tf.get_variable (name,shape)。
# name 是一个唯一标识这个变量对象的字符串。它在全局图中必须是唯一的，所以要确保不会出现重复的名称。
# shape 是一个与张量形状相对应的整数数组，它的语法很直观——每个维度对应一个整数，并按照排列。
# 例如，一个 3*8 的矩阵可能具有形状 [3,8]。要创建标量，请使用空列表作为形状：[]。
# count_variable = tf.get_variable('name', [])
# zero_node = tf.constant(0.)
# assign_node = tf.assign(count_variable, zero_node) # tf.assign (target,value) 声明节点赋值
# sess = tf.Session()
# # print(sess.run(count_variable)) # 这句报错
# print(sess.run('name'))
# print(sess.run(assign_node))
# print(sess.run(count_variable))
# print(sess.run('name'))


# 带变量初始化方法2
# const_init_node = tf.constant_initializer(0.)
# count_variable = tf.get_variable('name', [2, 3], initializer=const_init_node)  # get_variable加上初始化参数
# init = tf.global_variables_initializer()  # 再创建一个全局带init变量的初始化器
# sess = tf.Session()
# sess.run(init)  # run这个初始化器，就能初始化全部带initializer的get_variable，没这句会报错
# print(sess.run(count_variable))


# 计算线性回归 y=mx+b
m = tf.get_variable('m', [], initializer=tf.constant_initializer(0.))
b = tf.get_variable('b', [], initializer=tf.constant_initializer(0.))
init = tf.global_variables_initializer()

input_placeholder = tf.placeholder(tf.float32)
output_placeholder = tf.placeholder(tf.float32)

x_learned = input_placeholder   # 训练集里的x
y_learned = output_placeholder  # 训练集里的y
y_guess = m * x_learned + b     # 猜的的y
loss = tf.square(y_guess - y_learned)  # 两个y的差的平方

# 固定学习率
# train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 这句定义训练目标node，
# GradientDescentOptimizer指的是梯度下降算法，每次训练会更新mb的值
# 0.001是学习率，即m和b每次试算的变化量是多少
# 后面是训练目标，即loss最小（minimize）
# 这里学习率取的是固定值 0.001，也可以写成下面这样，先写一个大的学习率，学习率随着训练次数增加，变化量越来越小
# learning_rate=0.001
# groable=tf.Variable(tf.constant(0))
# lrate=tf.train.exponential_decay(learning_rate,groable,100,0.89)
# 完整函数：
# tf.train.exponential_decay( learning_rate,初始学习率
# global_step,当前迭代次数
# decay_steps,衰减速度（在迭代到该次数时学习率衰减为earning_rate * decay_rate）
# decay_rate,学习率衰减系数，通常介于0-1之间。
# staircase=False,(默认值为False,当为True时，（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。
# name=None )
# 学习率会按照以下公式变化
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

# 递减学习率
learning_rate = 0.1  # 需要配合下面10000和0.8这三个数进行调整
groable= tf.get_variable('groable', [], initializer=tf.constant_initializer(0.))
lrate=tf.train.exponential_decay(learning_rate, groable, 10000, 0.8)  # decay_steps 样本的1/3到1/5左右
train_op = tf.train.GradientDescentOptimizer(lrate).minimize(loss, global_step=groable)
init = tf.global_variables_initializer()

# 然后我们随机定义一套mb，看训练出来的结果和随机数差多少
true_m = random.random()
true_b = random.random()

sess = tf.Session()
sess.run(init)

# 自己随机一套训练集
for i in range(50000):
    x = random.random()
    y = true_m * x + true_b

    # 对于每一个训练集进行训练
    _loss, _ = sess.run([loss, train_op], feed_dict={input_placeholder: x, output_placeholder: y})
    if i % 5000 == 0:
        print(i, _loss)
        print('learn parameters:   m = %.4f, b = %.4f' % (tuple(sess.run([m, b]))))

print('true parameters:    m = %.4f, b = %.4f' % (true_m, true_b))

