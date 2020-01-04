# 这里是’tensorflow’的基本知识
-----
## 一、tensorflow的数据类型
**tf.Tensor是相当于np.array的存在**
---

### 在tensorflow里面，有以下四种数据类型
   + scalar，标量，就是单个的数，比如，1，2，4，这类
   + vector，向量，[1，2，3]
   + matrix，矩阵，[[1,2,3],[1,2,3]]
   + tensor,张量，dim>2时就把它统称为tensor
## 二、tensor的创建
---
   + 利用numpy转换 tf.convert_to_tensor
   + 利用tensor本身的函数创建，ones、zeros、fill、random
   + tf.constant()
   ---
   **将tensorflow的数据类型一一对应到神经网络里面**
   + saclar对应loss
   + vector 对应bias
   +tensor 对应input、w、表示文本、表示图像
