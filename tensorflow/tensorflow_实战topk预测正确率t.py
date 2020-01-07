import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2467)

# 计算正确预测率
def accuracy(output, target, topk=(1,)):
    # 可以随意设置top_k的数值，用maxk来接收
    # bitch_size统计总共的预测数用以计算正确率
    maxk = max(topk)
    batch_size = target.shape[0]

    # pred为top_K的输出索引转置成从上至下的排列
    # target用broadcast转化成一样的大小便于比较
    # 先计算出 maxk 的所有情况得出topn的矩阵
    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)

    # 依次计算topk的正确率
    # 第topk的正确率等于取前n行的矩阵，计算预测对的总值/预测的总值
    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype = tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k*(100.0/batch_size))
        res.append(acc)
    return res


# 用random的方式生成预测值，softmax限制每一行的和为1
# 用uniform的方式生成正确值，刚好与索引值对应
# 打印出可能值
# 打印出预测值
# 打印出实际值
# 打印topk的正确率
output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype = tf.int32)
print('preb:', output.numpy())
print('target:', target.numpy())

acc = accuracy(output, target, topk=(1, 2, 3, 4, 5, 6))
print('top-1-6:', acc)
