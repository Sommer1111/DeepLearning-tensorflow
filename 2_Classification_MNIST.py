import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

# 只显示错误和警告
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 将二维的数据实现降维，[28，28，1]一张图片的相素点矩阵转化成[28*28,1]的一维向量；
# 将一维向量继续降，利用激活函数转化成非线性的关系，映射到十维的向量，按概率分成十类，概率大的为类
# 自动从网上下载数据集
(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # 将数据集中的numpy类型的转化成 tensorflow 格式
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 将转化的数据切片，一次算一部分
train_dataset = train_dataset.batch(200)



# 运用Dese函数，直接进行降维操作，三层激励函数
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])
# optimizers直接实现 w' = w-lr*gradient 的更新过程
optimizers = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):

    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28*28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        grads = tape.gradient(loss, model.trainable_variables)
        optimizers.apply_gradients(zip(grads, model.trainable_variables))
        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
