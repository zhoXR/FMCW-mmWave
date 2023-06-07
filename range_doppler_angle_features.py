import numpy as np
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt
from utils.movieMaker import movieMaker
from utils.forceAspect import forceAspect

from utils.SignalParams import SignalParams
from utils.get_range_feature import get_range_feature
from utils.get_doppler_feature import get_doppler_feature

params = SignalParams()

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
filepath = dirname + filename + '_Raw_0.bin'
adc_data = np.fromfile(filepath, dtype=np.int16)
adc_data = adc_data.reshape(params.numFrames,
                            -1)  # size (numFrames, numTxAntennas*numRxAntennas*numLoopsPerFrame*numADCSamples)
adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=params.numChirpsPerFrame,
                               num_rx=params.numRxAntennas,
                               num_samples=params.numADCSamples)  # size (numFrames, numTxAntennas*numLoopsPerFrame, numRxAntennas, numADCSamples)
print("Data Loaded!")

# -----------------------------------------------------------------
# -------------------------- range feature ------------------------
# -----------------------------------------------------------------
range_data, micro_range_data, range_angle_data = get_range_feature(adc_data, params)
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

# -----------------------------------------------------------------
# -------------------------- doppler feature ----------------------
# -----------------------------------------------------------------
micro_doppler_data, doppler_angle_data = get_doppler_feature(adc_data, params)
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

# -----------------------------------------------------------------
# -------------------------- angle feature ------------------------
# -----------------------------------------------------------------
# plot range-angle per frame
range_angle_ims = []
range_angle_fig = plt.figure(dpi=300)
for i in range(params.numFrames):
    range_angle_ims.append((
        plt.imshow(np.squeeze(range_angle_data[i, :, :]).T, origin='lower',
                   extent=(-22, 22, params.ranges.min(), params.ranges.max())),
        plt.xlabel('Angle Bins'),
        plt.ylabel('Range (meters)'),
        plt.title("Range Angle Plot"),))
    forceAspect(range_angle_fig.gca(), aspect=1)
movieMaker(range_angle_fig, range_angle_ims, filename + '_range_angle_fig.mp4')

import matplotlib.pyplot as plt

# plot range-angle per frame
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
