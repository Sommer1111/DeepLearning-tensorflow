import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


# 数据预处理
def proprecess(x, y):
    x = tf.cast(x, dtype = tf.float32)/255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype = tf.int32)
    y = tf.one_hot(y, depth = 10)
    return x, y


# 数据加载
(x, y), (x_test, y_test) = datasets.mnist.load_data()

# 数据分批，迭代器
batchsz = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(proprecess).shuffle(60000).batch(batchsz)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(proprecess).batch(batchsz)

sample = next(iter(db))

# 神经网络构架
network = Sequential([
    layers.Dense(256, activation = 'relu'),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(10)
    ])
network.build(input_shape = [None, 28*28])
network.summary()

# 训练
network.compile(optimizer = optimizers.Adam(lr = 0.01),
                loss = tf.losses.CategoricalCrossentropy(from_logits =True),
                metrics=['accuracy']
                )

# 测试
network.fit(db, epochs = 3, validation_data = db_test, validation_freq = 2)

# 评估
network.evaluate(db_test)

# 保存数据
network.save_weights('weight.ckpt')
print('save weights')

# 模拟突发事件
del network

# 重新建立网络,训练网络
network = Sequential([
    layers.Dense(256, activation = 'relu'),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(10)
])

network.compile(optimizer = optimizers.Adam(lr = 0.01),
                loss = tf.losses.CategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy']
                )

# 加载保存的数据
network.load_weights('weight.ckpt')

# 测试性能
print('load weights')
network.evaluate(db_test)