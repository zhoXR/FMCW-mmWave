import mmwave.dsp as dsp
from mmwave.dsp.utils import Window
import numpy as np
import scipy.linalg as LA
from scipy import signal
import matplotlib.pyplot as plt

derad = np.pi / 180  # pi是圆周率
radeg = 180 / np.pi


def MUSIC(K, Y, n, SP):
    iwave = 1  # 发射天线的个数
    f = 77e9  # 频率为77GHz
    c = 3e8  # 光速
    wavelength = c/f  # 波长
    Angles = np.linspace(0, np.pi, 180)  # 在线性空间中以均匀步长生成数字序列
    d = np.arange(0, 8*wavelength/2, wavelength/2)  # 接收天线之间的距离
    Rxx = Y @ (Y.conj().T) / n  # Y*Y共轭转置
    D, EV = LA.eig(Rxx)  # 特征值 特征向量
    index = np.argsort(D)  # 从小到大排序
    EN = EV.T[index].T[:, 0:K - iwave]  # 用前面的几个小特征值的特征向量

    for i in range(180):
        a = np.exp(-1j * 2 * np.pi * d.reshape(-1, 1) * np.sin(Angles[i])/wavelength)  # 相位
        SP[i] = (1 / np.abs((a.conj().T @ EN @ EN.conj().T @ a)))
    # SP = np.abs(SP)
    # SPmax = np.max(SP)
    # SP = 10 * np.log10(SP / SPmax)
    x = Angles * radeg
    plt.plot(x, SP)
    plt.show()
    return SP


def get_range_feature(adc_data, params):
    sos1 = signal.iirfilter(25, [30, 70], rs=80, btype='band',
                            analog=False, ftype='cheby2', fs=160,  # analog=Fslse表示返回数字滤波器
                            output='sos')  # sos表示二进制
    sos2 = signal.iirfilter(25, [5, 30], rs=80, btype='band',
                            analog=False, ftype='cheby2', fs=255,
                            output='sos')

    range_data = np.zeros(
        (params.numFrames, params.numTxAntennas * params.numRxAntennas, params.numLoopsPerFrame, params.numADCSamples),
        dtype=np.complex_)
    micro_doppler_data = np.zeros(
        (params.numFrames, params.numTxAntennas * params.numRxAntennas, params.numLoopsPerFrame, params.numADCSamples),
        dtype=np.float64)
    aoa_input_data = np.zeros(
        (params.numFrames, params.numTxAntennas * params.numRxAntennas, params.numLoopsPerFrame, params.numADCSamples),
        dtype=np.complex_)
    range_angle_res = np.zeros((params.numFrames, params.numADCSamples, 180), dtype=np.float64)
    azimuth_fft = np.zeros((params.numADCSamples, 180), dtype=np.float64)

    for i, frame in enumerate(adc_data):
        frame_allVirtualAntenna = dsp.separate_tx(frame, params.numTxAntennas, vx_axis=1, axis=0)
        for virantennaIdx in range(params.numTxAntennas * params.numRxAntennas):
            frame_singleAntenna = np.expand_dims(frame_allVirtualAntenna[:, virantennaIdx, :], 1)

            # Range Processing
            range_res = dsp.range_processing(frame_singleAntenna, window_type_1d=Window.HANNING)  # 傅里叶变换
            range_data[i, virantennaIdx, :, :] = np.squeeze(range_res)

            range_res_filtered = signal.sosfiltfilt(sos2, np.squeeze(range_res), axis=0)  # 与x形状相同的过滤输出
            range_res = np.expand_dims(range_res_filtered, 1)

            # Doppler Processing
            det_matrix, aoa_input = dsp.doppler_processing(range_res, num_tx_antennas=1, clutter_removal_enabled=True,
                                                           window_type_2d=Window.HANNING, interleaved=True,
                                                           accumulate=True)

            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)  # 将零频点移到频谱的中间
            micro_doppler_data[i, virantennaIdx, :, :] = det_matrix_vis.T
            aoa_input_data[i, virantennaIdx, :, :] = np.squeeze(aoa_input).T
        # (4) Angle Processing
        aoa_input_data_shift = np.fft.fftshift(np.squeeze(aoa_input_data[i, 0:params.numRxAntennas * 2, :, :]),
                                               axes=1)
        for j in range(params.numADCSamples):
            aoa_input_data_ = np.squeeze(aoa_input_data_shift[:, :, j])
            SP = np.empty(180, dtype=complex)  # 按照给定维度生成数组 值不变
            azimuth_fft[j, :] = MUSIC(K=8, Y=aoa_input_data_, n=255, SP=SP)
        # azimuth_fft = MUSIC(K=8, d=d, theta=theta, snr=10, n=500, SP=SP)
        # azimuth_fft = np.fft.fft(aoa_input_data_shift, params.numAngleBins, axis=0)  # 快速傅里叶变换 (FFT) 计算，改变变换轴的长度
        # azimuth_fft = np.fft.fftshift(azimuth_fft, axes=0)
        # azimuth_fft_abs = np.abs(azimuth_fft)
        # range_angle = np.sum(azimuth_fft_abs, axis=1)
        range_angle_res[i, :, :] = azimuth_fft

    return range_data, micro_doppler_data, range_angle_res
