import numpy as np
import scipy.signal as ss
import scipy.linalg as LA
import matplotlib.pyplot as plt

derad = np.pi / 180  # pi是圆周率
radeg = 180 / np.pi


def awgn(x, snr):
    spower = np.sum((np.abs(x) ** 2)) / x.size
    x = x + np.sqrt(spower / snr) * (
            np.random.randn(x.shape[0], x.shape[1]) + 1j * np.random.randn(x.shape[0], x.shape[1]))
    return x


def MUSIC(K, d, theta, snr, n):
    iwave = theta.size
    A = np.exp(-1j * 2 * np.pi * d.reshape(-1, 1) @ np.sin(theta * derad))  # 发射源从三个角度发射的信号
    S = np.random.randn(iwave, n)  # 返回一个或一组样本，具有标准正态分布
    X = A @ S  # 生成接收到的信号
    X = awgn(X, snr)  # 加入白噪音，产生小特征值
    Rxx = X @ (X.conj().T) / n  # 得到协方差矩阵
    D, EV = LA.eig(Rxx)  # 特征值，特征向量
    index = np.argsort(D)  # 返回排序后的索引值的数组
    EN = EV.T[index].T[:, 0:K - iwave]

    for i in range(numAngles):
        a = np.exp(-1j * 2 * np.pi * d.reshape(-1, 1) * np.sin(Angles[i]))
        SP[i] = ((a.conj().T @ a) / (a.conj().T @ EN @ EN.conj().T @ a))[0, 0]
    return SP


Angles = np.linspace(-np.pi/2, np.pi/2, 360)  # 在线性空间中以均匀步长生成数字序列
numAngles = Angles.size
d = np.arange(0, 4, 0.5)  # 接收天线之间的距离
theta = np.array([10, 30, 60]).reshape(1, -1)  # ？
SP = np.empty(numAngles, dtype=complex)  # 按照给定维度生成数组 值不变
SP = MUSIC(K=8, d=d, theta=theta, snr=10, n=500)

SP = np.abs(SP)
SPmax = np.max(SP)
SP = 10 * np.log10(SP / SPmax)
x = Angles * radeg
plt.plot(x, SP)
plt.show()
