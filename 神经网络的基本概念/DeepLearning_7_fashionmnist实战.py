import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 数据加载
(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()


# 数据预处理
def preprocess(x, y):
    x = tf.cast(x, dtype = tf.float32)/255.
    y = tf.cast(y, dtype = tf.int32)
    return x, y


# 数据分批，预处理打乱，迭代循环
batchsz = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

db_iter = iter(db)
sample = next(db_iter)
# sample[0]表示x迭代的大小，sample代表y迭代的大小
print('batch:', sample[0].shape, sample[1].shape)


# 数据优化，更新梯度, w = w-lr*grad
optimizer = optimizers.Adam(lr=1e-3)



# 神经网络层搭建
# Squential的作用是创建一个神经网络的框架
model = Sequential([
    layers.Dense(256, activation = tf.nn.relu),
    layers.Dense(128, activation = tf.nn.relu),
    layers.Dense(64, activation = tf.nn.relu),
    layers.Dense(32, activation = tf.nn.relu),
    layers.Dense(10)
])
# Squential建立好神经网络的框架之后需要喂数据
# summary的作用是打印神经网络结构便于查看
model.build(input_shape=[None, 28*28])
model.summary()


# 训练,训练多少次？每一次的循环里面根据batch迭代
# 训练的过程，x转化为一维，在特定的with里面记录graident
# 预测值logits，y转化为onehot对应标准答案，MSE、mean，计算之间的loss
# 计算loss对w、b的梯度，model.trainable_variables大口袋接收所有要求导的w、b
# 优化器，optimizer.apply_graduents更新梯度,w = w- lr*grad，一一对应
def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            x= tf.reshape(x, [-1, 28*28])
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth = 10)
                loss = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        # 测试
        # 测试集同等处理，y经softmax、argmax、cast转化成概率最大的索引即为其分类
        # 比较是否与答案相等，记录算对的总数，记录预测的总数，得到正确率

        total_correct = 0
        total_num = 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28*28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis = 1)
            prob = tf.argmax(prob, axis = 1)
            pred = tf.cast(prob, dtype = tf.int32)

            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)


# 程序运行
if __name__ == '__main__':
    main()
