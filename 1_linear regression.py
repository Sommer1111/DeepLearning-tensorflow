# 只用numpy模块简单实现线性回归
import numpy as np


# 计算出损失函数
def computer_error_numpy(b,w,points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]

        totalError += (y - (w * x + b ))**2
    return totalError / float(len(points))


# 得到一次所有数据的计算出的梯度值
def learn_gradient(b_current, w_current, points, learningrate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        w_gradient += (2/N) * ((w_current * x + b_current) - y)*x

    b_now = b_current-(learningrate*b_gradient)
    w_now = w_current-(learningrate*w_gradient)
    return [b_now, w_now]


# 进行循环更新梯度
def gradient_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w

    for i in range(num_iterations):
        b, w = learn_gradient(b, w, np.array(points), learning_rate)
    return[b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    learning_rate = 0.0001 # 这里有关学习率的选取比较奇怪。当我设置为0.001时，提示迭代最后出现了全为none的情况
    [b, w] = gradient_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 computer_error_numpy(b, w, points))
          )


if __name__ == '__main__':
    run()
