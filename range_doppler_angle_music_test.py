import numpy as np
import sci as sci

from mmwave.dataloader import DCA1000
from utils.SignalParams import SignalParams
from utils.get_doppler_feature import get_doppler_feature
from utils.get_range_feature_music import get_range_feature


def prepare_data():
    """
数据导入并处理维度
    """
    dirname = 'E:/ZXR/data/rawdata/'
    filename = '04_zy_01'
    filepath = dirname + '\\' + filename + '_Raw_0.bin'
    data = np.fromfile(filepath, dtype=np.int16)  # 将采集的数据导入并转换为16进制
    data = data.reshape(params.numFrames, -1)  # 将数据拆分成130帧

    # 将数据拆分成130*765*4*80，分别代表帧、3根接收天线收到的信号（3*255=765）每帧255个chirp、4个发射天线、每个chirp采样80个点
    data = np.apply_along_axis(DCA1000.organize, 1, data, num_chirps=params.numChirpsPerFrame,
                               num_rx=params.numRxAntennas,
                               num_samples=params.numADCSamples)
    return data


if __name__ == '__main__':
    filename = '01_zxr_90'
    datapath = r'E:\ZXR\data\rawdata'
    params = SignalParams()
    adc_data = prepare_data()
    range_data, micro_range_data, range_angle_data = get_range_feature(adc_data, params)
    micro_doppler_data, doppler_angle_data, aoa_input = get_doppler_feature(adc_data, params)
    range_time_plot = np.sum(micro_range_data, 2)
    range_time = np.sum(range_time_plot, 1).T
    doppler_time_plot = np.sum(micro_doppler_data, 3)
    doppler_time = np.sum(doppler_time_plot, 1).T
    sci.savemat(datapath + '\\' + filename + '.mat',
                {'range_data': range_time, 'doppler_data': doppler_time})
