import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

#  只显示error，不显示其他无用信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#  自动检测并载入数据集,x与y的大小如下
#  x:[60k,28,28]
#  y:[60k]
(x, y), _ = datasets.mnist.load_data()
#  将x的范围归一化x:[0~255]-->[0~1],并且有numpy转化成tensor的格式
x = tf.convert_to_tensor(x, dtype=tf.float32)/255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# 将60k的数据集分成很多个大小为128的batch,同时得到迭代次数
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
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

# 打印结果
# (60000, 28, 28) (60000,) <dtype: 'float32'> <dtype: 'int32'>
# tf.Tensor(0.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32)
# tf.Tensor(0, shape=(), dtype=int32) tf.Tensor(9, shape=(), dtype=int32)

# batch (128, 28, 28) (128,)
# 0 0 loss 0.7764955163002014
# 0 100 loss 0.3396267294883728
# 0 200 loss 0.3010321259498596
# 0 300 loss 0.2663792073726654
# 0 400 loss 0.23309285938739777
# 1 0 loss 0.23417365550994873
# 1 100 loss 0.2222554236650467
# 1 200 loss 0.20121845602989197
# 1 300 loss 0.1904672384262085
# 1 400 loss 0.17274245619773865
# 2 0 loss 0.17522092163562775
# 2 100 loss 0.1783313751220703
# 2 200 loss 0.1606927216053009
# 2 300 loss 0.15727169811725616
# 2 400 loss 0.14462289214134216
# 3 0 loss 0.1461002081632614
# 3 100 loss 0.1544061005115509
# 3 200 loss 0.13865795731544495
# 3 300 loss 0.1383732259273529
# 3 400 loss 0.12777754664421082
# 4 0 loss 0.12837953865528107
# 4 100 loss 0.13878782093524933
# 4 200 loss 0.12437250465154648
# 4 300 loss 0.12577030062675476
# 4 400 loss 0.11616740375757217
# 5 0 loss 0.11619359254837036
# 5 100 loss 0.12750890851020813
# 5 200 loss 0.11410540342330933
# 5 300 loss 0.11656177043914795
# 5 400 loss 0.1075325459241867
# 6 0 loss 0.1071401983499527
# 6 100 loss 0.11881832778453827
# 6 200 loss 0.10623922199010849
# 6 300 loss 0.10940057039260864
# 6 400 loss 0.1007702499628067
# 7 0 loss 0.10008988529443741
# 7 100 loss 0.11183450371026993
# 7 200 loss 0.099940225481987
# 7 300 loss 0.10360991954803467
# 7 400 loss 0.09530206769704819
# 8 0 loss 0.09440962225198746
# 8 100 loss 0.10603219270706177
# 8 200 loss 0.09474048763513565
# 8 300 loss 0.09879303723573685
# 8 400 loss 0.09077998250722885
# 9 0 loss 0.08972539752721786
# 9 100 loss 0.10113056004047394
# 9 200 loss 0.09034091234207153
# 9 300 loss 0.09469185769557953
# 9 400 loss 0.08697400987148285
#
# Process finished with exit code 0
