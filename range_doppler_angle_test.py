import numpy as np
import sci as sci
import matplotlib.pyplot as plt
from utils.movieMaker import movieMaker
from utils.forceAspect import forceAspect
from mmwave.dataloader import DCA1000
from utils.SignalParams import SignalParams
from utils.get_doppler_feature import get_doppler_feature
from utils.get_range_feature_music_test import get_range_feature


def prepare_data():
    """
数据导入并处理维度
    """
    filepath = datapath + '\\' + filename + '_Raw_0.bin'
    data = np.fromfile(filepath, dtype=np.int16)  # 将采集的数据导入并转换为16进制
    data = data.reshape(params.numFrames, -1)  # 将数据拆分成130帧

    # 将数据拆分成130*765*4*80，分别代表帧、3根接收天线收到的信号（3*255=765）每帧255个chirp、4个发射天线、每个chirp采样80个点
    data = np.apply_along_axis(DCA1000.organize, 1, data, num_chirps=params.numChirpsPerFrame,
                               num_rx=params.numRxAntennas,
                               num_samples=params.numADCSamples)
    return data


def plot_range_time():
    range_time_plot = np.sum(np.log10(np.abs(range_data)), 2)
    range_time_fig = plt.figure(figsize=(15, 12), dpi=300)
    for virantennaIdx in range(params.numTxAntennas * params.numRxAntennas):
        range_time_sinAntenna = np.squeeze(range_time_plot[:, virantennaIdx, :]).T
        plt.subplot(3, 4, virantennaIdx + 1)
        plt.imshow(range_time_sinAntenna, origin='lower',
                   extent=(params.times.min(), params.times.max(), params.ranges.min(), params.ranges.max()))
        plt.xlabel('Times(s)')
        plt.ylabel('Range (meters)')
        plt.title("Range time Plot")
    plt.show()


def plot_doppler_time():
    doppler_time_plot = np.sum(micro_doppler_data, 3)
    doppler_time_fig = plt.figure(figsize=(16, 13), dpi=300)
    for virantennaIdx in range(params.numTxAntennas * params.numRxAntennas):
        doppler_time_sinAntenna = np.squeeze(doppler_time_plot[:, virantennaIdx, :]).T
        plt.subplot(3, 4, virantennaIdx + 1)
        plt.imshow(doppler_time_sinAntenna, origin='lower',
                   extent=(params.times.min(), params.times.max(), params.velocities.min(), params.velocities.max()))
        forceAspect(doppler_time_fig.gca(), aspect=1)
        plt.xlabel('Times(s)')
        plt.ylabel('Velocity (meters per second)')
        plt.title("Doppler time Plot")
    plt.show()


def plot_range_angle():
    range_angle_ims = []
    range_angle_fig = plt.figure(dpi=300)
    for i in range(1):
        range_angle_ims.append((
            plt.imshow(np.squeeze(range_angle_data[i, :, :]).T, origin='lower',
                       extent=(0, 180, params.ranges.min(), params.ranges.max())),
            plt.xlabel('Angle Bins'),
            plt.ylabel('Range (meters)'),
            plt.title("Range Angle Plot"),))
        forceAspect(range_angle_fig.gca(), aspect=1)  # 设置横纵比
    # movieMaker(range_angle_fig, range_angle_ims, datapath + filename + '_range_angle_fig.gif')
    plt.show()


def plot_range_doppler():
    range_doppler_ims = []
    range_doppler_fig = plt.figure(dpi=300)
    for i in range(1):
        range_doppler_ims.append((
            plt.imshow(np.squeeze(range_doppler_data[i, :, :]).T, origin='lower',
                       extent=(params.velocities.min(), params.velocities.max(), params.ranges.min(), params.ranges.max())),
            plt.xlabel('Velocity (meters per second)'),
            plt.ylabel('Range (meters)'),
            plt.title("Range doppler Plot"),))
        forceAspect(range_doppler_fig.gca(), aspect=1)  # 设置横纵比
    # movieMaker(range_angle_fig, range_angle_ims, datapath + filename + '_range_angle_fig.gif')
    plt.show()


def plot_doppler_angle():
    doppler_angle_ims = []
    doppler_angle_fig = plt.figure(dpi=300)
    for i in range(params.numFrames):
        doppler_angle_ims.append((
            plt.imshow(np.squeeze(doppler_angle_data[i, :, :]).T, origin='lower',
                       extent=(-22, 22, params.velocities.min(), params.velocities.max())),
            plt.xlabel('Angle Bins'),
            plt.ylabel('Velocity (meters per second)'),
            plt.title("Doppler Angle Plot"),))
        forceAspect(doppler_angle_fig.gca(), aspect=1)
    movieMaker(doppler_angle_fig, doppler_angle_ims, filename + '_doppler_angle_fig.mp4')


if __name__ == '__main__':
    filename = '01_zxr_90'
    datapath = 'E:/ZXR/data/zxr/'
    params = SignalParams()
    adc_data = prepare_data()
    range_data, micro_range_data, range_angle_data = get_range_feature(adc_data, params)
    micro_doppler_data, doppler_angle_data, aoa_input = get_doppler_feature(adc_data, params)
    plot_range_time()
    plot_doppler_time()
    plot_range_angle()
    plot_doppler_angle()

    range_time_plot = np.sum(micro_range_data, 2)
    range_time = np.sum(range_time_plot, 1).T
    doppler_time_plot = np.sum(micro_doppler_data, 3)
    doppler_time = np.sum(doppler_time_plot, 1).T
    sci.savemat(datapath + '\\' + filename + '.mat',
                {'range_data': range_time, 'doppler_data': doppler_time})
