# 这里是<span style="color:red">`tensorflow`的基本知识</span>.
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

---
## 四、`tensorflow`的高阶操作
   + [合并与分割](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E5%90%88%E5%B9%B6%E4%B8%8E%E5%88%86%E5%89%B2.ipynb)
   + [数据统计](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E6%95%B0%E6%8D%AE%E7%BB%9F%E8%AE%A1.ipynb)
   + [张量排序](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E5%BC%A0%E9%87%8F%E6%8E%92%E5%BA%8F.ipynb)
   + [填充与复制](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E5%A1%AB%E5%85%85%E4%B8%8E%E5%A4%8D%E5%88%B6.ipynb)
   + [数据限幅](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E6%95%B0%E6%8D%AE%E9%99%90%E5%B9%85.ipynb)
   + [高阶操作](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E9%AB%98%E9%98%B6%E6%93%8D%E4%BD%9C.ipynb)
   + [数据加载](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE%E9%9B%86.ipynb)
   + [**测试实战**](https://github.com/Sommer1111/DeepLearning-pycharm/blob/master/tensorflow/tensorflow_forward_war_test.py)
