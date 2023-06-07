import numpy as np
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt
import scipy.io as sci
from utils.movieMaker import movieMaker
from utils.forceAspect import forceAspect

from utils.SignalParams import SignalParams
from utils.get_range_feature import get_range_feature
from utils.get_doppler_feature import get_doppler_feature

params = SignalParams()
font1 = {'weight': 60, 'size': 50}
# -----------------------------------------------------------------
# -------------------------- 信号参数 -----------------------------
# -----------------------------------------------------------------
print(f'Time Resolution: {params.framePeriod} [seconds] / Time Max: {params.times.max()} [seconds]')
print(f'Range Resolution: {params.range_resolution} [meters] / Range Max: {params.ranges.max()} [meters]')
print(f'Velocity Resolution: {params.doppler_resolution} [meters/second] / Velocity Max: {params.velocities.max()} [meters/second]')

# -----------------------------------------------------------------
# -------------------------- 导入数据并整型 -------------------------
# -----------------------------------------------------------------

dirname = 'E:/ZXR/data/rawdata/'
filename = '01_zy_01'
filepath = dirname + '\\' + filename + '_Raw_0.bin'
adc_data = np.fromfile(filepath, dtype=np.int16)
adc_data = adc_data.reshape(params.numFrames,
                            -1)  # size (numFrames, numTxAntennas*numRxAntennas*numLoopsPerFrame*numADCSamples)
adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=params.numChirpsPerFrame,
                               # func是我们写的一个函 axis表示函数func对arr是作用于行还是列 arr便是我们要进行操作的数组了
                               num_rx=params.numRxAntennas,
                               num_samples=params.numADCSamples)  # size (numFrames, numTxAntennas*numLoopsPerFrame, numRxAntennas, numADCSamples)
print("Data Loaded!")

dopsavepath = 'dop_before_' + filename + '.pdf'
rangesavepath = 'range_before_' + filename + '.pdf'
datapath = r'D:\good good study\My project\ASL\py37\mmwave_gesture\data_process_kx2\range-doppler-compare'
# -----------------------------------------------------------------
# -------------------------- range feature ------------------------
# -----------------------------------------------------------------
range_data, micro_range_data, range_angle_data = get_range_feature(adc_data, params)

######################################### second range-time method
range_time_plot = np.sum(micro_range_data, 2)
range_time = np.sum(range_time_plot, 1).T

# -----------------------------------------------------------------
# -------------------------- doppler feature ----------------------
# -----------------------------------------------------------------
micro_doppler_data, doppler_angle_data, aoa_input = get_doppler_feature(adc_data, params)
doppler_time_plot = np.sum(micro_doppler_data, 3)
doppler_time = np.sum(doppler_time_plot, 1).T
sci.savemat(datapath + '\\' + filename + '.mat',
            {'range_data': range_time, 'doppler_data': doppler_time})
