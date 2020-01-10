import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras


# 加载数据
(x, y), (x_val, y_val) = datasets.cifar10.load_data()

# 数据预处理
def proprecess(x, y):
    x = 2*tf.cast(x, dtype = tf.float32)/255.-1.
    y = tf.cast(y, dtype = tf.int32)
    return x, y

# 数据处理，分批，迭代
# y[50k,1,10]==>[50k,10]
y = tf.squeeze(y)
y = tf.one_hot(y, depth = 10)
y_val = tf.squeeze(y_val)
y_val = tf.one_hot(y_val, depth = 10)

batchsz = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(proprecess).shuffle(10000).batch(batchsz)
db_test = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_test = db_test.map(proprecess).batch(batchsz)

sample = next(iter(db))

# 自定义网络层
class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x

class  MyNet(keras.Model):
     def __init__(self):
         super(MyNet, self).__init__()

         self.fc1 = MyDense(32*32*3, 256)
         self.fc2 = MyDense(256, 128)
         self.fc3 = MyDense(128, 64)
         self.fc4 = MyDense(64, 32)
         self.fc5 = MyDense(32, 10)

     def call(self, inputs, training=None):

         x = tf.reshape(inputs, [-1, 32*32*3])
         x = self.fc1(x)
         x = tf.nn.relu(x)
         x = self.fc2(x)
         x = tf.nn.relu(x)
         x = self.fc3(x)
         x = tf.nn.relu(x)
         x = self.fc4(x)
         x = tf.nn.relu(x)
         x = self.fc5(x)

         return x

# 训练网络
network = MyNet()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# 测试网络
network.fit(db_test, epochs=15, validation_data=db_test, validation_freq=1)




# 评估网络
network.evaluate(db_test)


# 保存weights
network.save_weights('save/weights.ckpt')
print('saved training weights')
# 模拟突发情况
del network

# 恢复数据
network = MyNet()
network.compile(optimizer = optimizers.Adam(lr = 1e-3),
                loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy'])
network.load_weights('save/weights.ckpt')
print('load weights')

# 评估网络
network.evaluate(db_test)