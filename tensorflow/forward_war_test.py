import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

#  只显示error，不显示其他无用信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#  自动检测并载入数据集,x与y的大小如下
#  x:[60k,28,28]
#  y:[60k]
(x, y), (x_test,y_test) = datasets.mnist.load_data()
#  将x的范围归一化x:[0~255]-->[0~1],并且有numpy转化成tensor的格式
x = tf.convert_to_tensor(x, dtype=tf.float32)/255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)/255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# 将60k的数据集分成很多个大小为128的batch,同时得到迭代次数
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch', sample[0].shape, sample[1].shape)

# [b,784]-->[b,256]-->[b,128]-->[b,10]
# w = [din_in, dim_out], b = [dim_out]

# 截断的随机正态分布取初值，但是这里要注意的是方差的取值，可能会导致梯度爆炸
# 虽然Variable的本质与tensor一致，但只有Variable可以记录梯度信息
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

# 一次循环，要迭代每一个bitch，进行多次循环训练网络
for epoch in range(10):
    for step, (x,y) in enumerate(train_db):
        # 三层非线性不断进行压缩
        # [b,28,28]-->[b,28*28]-->[b,256]-->[b,128]-->[b,10]
        x = tf.reshape(x, [-1, 28*28])
        # 在with里面记录梯度信息
        with tf.GradientTape() as tape:
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h1 = tf.nn.relu(h2)
            out = h2 @ w3 + b3

            # computer loss
            # out:[b,10],所以利用onehot将y[10]-->[b,10]
            y_onehot = tf.one_hot(y, depth = 10)
            loss = tf.square(y_onehot-out)
            loss = tf.reduce_mean(loss)
        #computer gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # w1=w1-lr*w1_grad
        # 因为进行计算之后会返回tensor数据类型，所以需要用assign更新为Variable类型
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step%100 ==0:
            print(epoch, step, 'loss', float(loss))

    total_correct , total_num = 0, 0
    for step, (x, y) in enumerate(test_db):
        # x:[b,28,28]=>[b,28*28]
        # out:[b,28*28]=>[b,256]=>[b,128]=>[b,10]
        x = tf.reshape(x,[-1, 28*28])
        h1 = tf.nn.relu(x@w1 + b1)
        h2 = tf.nn.relu(h1@w2 + b2)
        out = h2@w3 + b3
        # out[b,10]~R
        # prob:[b,10]~[0,1]
        # pred:[b,10]=>[b]
        # 将得到的输出out映射到[0,1]之间
        # 通过argmax得到概率最大的索引，用cast转化成int型，即为它预测的分类值
        prob = tf.nn.softmax(out, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype = tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct+=int(correct)
        total_num +=x.shape[0]

    acc = total_correct/total_num
    print('total acc:', acc)

