# 这里是`tensorflow`的基本知识
-----
## 一、`tensorflow`的数据类型
**`tf.Tensor`是相当于`np.array`的存在**
---

### 在`tensorflow`里面，有以下四种数据类型
   + `scalar`，标量，就是单个的数，比如，1，2，4，这类
   + `vector`，向量，[1，2，3]
   + `matrix`，矩阵，[[1,2,3],[1,2,3]]
   + `tensor`,张量，dim>2时就把它统称为tensor
## 二、`tensor`的创建
---
   + 利用numpy转换 tf.convert_to_tensor
   + 利用tensor本身的函数创建，ones、zeros、fill、random
   + tf.constant()
   ---
   **将`tensorflow`的数据类型一一对应到神经网络里面**
   + saclar对应loss
   + vector 对应bias
   +tensor 对应input、w、表示文本、表示图像
   
   ---
## 三、`tensorflow`的初阶操作
   + [基本数据类型](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B.ipynb)
   + [创建tensor](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E5%88%9B%E5%BB%BAtensor.ipynb)
   + [索引与切片](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E7%B4%A2%E5%BC%95%E5%88%87%E7%89%87.ipynb)
   + [维度变换](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E7%BB%B4%E5%BA%A6%E5%8F%98%E6%8D%A2.ipynb)
   + [数值运算](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E6%95%B0%E5%80%BC%E8%BF%90%E7%AE%97.ipynb)
   + [broadcast](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_broadcast.ipynb)
   + [**实战前向传播算法forward**](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E5%AE%9E%E6%88%98forward_test.py)
