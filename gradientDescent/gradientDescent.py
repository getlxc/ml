import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\\windows\\fonts\\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题


# 加载数据
def loadtxtAndcsv_data(fileName, split, dataType):  # 参数名、分割标志、数据类型
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)

# 画三维数据图
def plot_data(X):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax.set_zlabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('X2', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X1', fontdict={'size': 15, 'color': 'red'})
    plt.show()


# 画每次迭代代价的变化图
def plotJ(history, num_iters):
    x = np.arange(1, num_iters + 1)
    plt.plot(x, history)
    plt.xlabel(u"迭代次数", fontproperties=font)  # 注意指定字体，要不然出现乱码问题
    plt.ylabel(u"代价值", fontproperties=font)
    plt.title(u"代价随迭代次数的变化", fontproperties=font)
    plt.show()


# 归一化feature
def featureNormaliza(X):
    X_norm = np.array(X)  # 将X转化为numpy数组对象，才可以进行矩阵的运算
    # 定义所需变量
    mu = np.mean(X_norm, 0)  # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X_norm, 0)  # 求每一列的标准差
    for i in range(X.shape[1]):  # 遍历列
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]  # 归一化
    return X_norm, mu, sigma


def computerCost(X, y, theta):
    # X是一个m行、n列的矩阵。y是一个m行1列的矩阵，theta是 1 行，n 列的矩阵。
    # 计算代价，主要用于画图时模拟一下。
    # 实现对应元素相乘，有2种方式，一个是np.multiply()，另外一个是 *
    a = (np.transpose(X * theta - y))
    b = (X * theta - y)
    c = a * b
    return c / (2 * len(y))


# 梯度下降算法
def gradientDescent(X, y, theta, alpha, num_iters):
    # X是一个m行3列的一个矩阵、y是每一个Xi的函数值、theta是权值、alpha是每一次的步长（学习率）、num是迭代的次数
    m = len(y)
    n = len(theta)
    history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值
    for i in range(num_iters):  # 遍历迭代次数
        h = np.dot(X, theta)  # 计算内积，matrix可以直接乘
        a = ((alpha / m) * (np.dot(np.transpose(X), h - y)))
        theta = np.mat(theta - a)  # 梯度的计算
        history[i] = computerCost(X, y, theta)  # 调用计算代价函数
        if (i % 100000 == 0):
            print('总共', num_iters, ' 当前迭代了', i / num_iters * 100, '%了')

    return theta, history


def linearRegression(data, alpha, num_iters, isUnification):
    X = data[:, 0:-1]  # X是前n-1列，即自变量的列
    y = data[:, -1]  # y对应最后一列，即函数值
    m = len(y)  # 总的数据条数
    col = data.shape[1]  # data的列数
    if isUnification:
        X, mu, sigma = featureNormaliza(X)  # 归一化 均值mu,标准差sigma,

    if X.shape[1]<=2 and X.shape[1]>=1:
        plot_data(data)  # 绘制数据的归一化效果

    X = np.hstack((np.ones((m, 1)), X))  # 在X前加一列1,这个是为了表示后面那个常数的，*1还是等于原来的数
    theta = np.zeros((col, 1))  # 初始化权值向量
    y = y.reshape(-1, 1)  # 将行向量转化为列
    theta, history = gradientDescent(X, y, theta, alpha, num_iters)
    plotJ(history, num_iters)
    return theta  # 返回和学习的结果theta


# 测试学习效果（预测）
def predict(mu, sigma, theta):  # 每一列的平均值，平均差，权值
    predict = np.array([1650, 3])
    norm_predict = (predict - mu) / sigma
    final_predict = np.hstack((np.ones((1)), norm_predict))
    result = np.dot(final_predict, theta)  # 预测结果
    return result


if __name__ == "__main__":

    isUnification = False  # 是否归一化, 归一化后 测试数据也要归一化才能进行预测。
    alpha = 0.000001  # 学习率
    num_iter = 50000  # 迭代次数

    data = loadtxtAndcsv_data("data.txt", ",", np.float64)  # 读取数据
    theta = linearRegression(data, alpha, num_iter, isUnification)

    for i in range(len(theta)):  # 输出权值
        print('权值', i, '=', theta[i][0])
