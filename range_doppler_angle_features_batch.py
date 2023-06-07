import os
import numpy as np
import scipy.io as sci
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
print(
    f'Velocity Resolution: {params.doppler_resolution} [meters/second] / Velocity Max: {params.velocities.max()} [meters/second]')

# -----------------------------------------------------------------
# -------------------------- 导入数据并整型 -------------------------
# -----------------------------------------------------------------
dirname = 'E:/ZXR/data/rawdata/'
gestures = 1
persons = {'zy'}
conditions = [1]
counts = 5
img_dir = 'E:/ZXR/data/rawdata/'
for gesId in range(gestures):
    for person in persons:
        for conId in conditions:
            # for timeId in range(counts):
                filename = str(gesId + 1).zfill(2) + '_' + person + '_' + str(
                    conId).zfill(2)
                filepath = dirname + filename + '_Raw_0.bin'
                if not os.path.exists(filepath):
                    print(filepath + ' is not exist!')
                else:
                    adc_data = np.fromfile(filepath, dtype=np.int16)
                    adc_data = adc_data.reshape(params.numFrames,
                                                -1)
                    # size (numFrames, numTxAntennas*numRxAntennas*numLoopsPerFrame*numADCSamples)
                    adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=params.numChirpsPerFrame,
                                                   num_rx=params.numRxAntennas,
                                                   num_samples=params.numADCSamples)
                    # size (numFrames, numTxAntennas*numLoopsPerFrame, numRxAntennas, numADCSamples)
                    print("Data Loaded!")

                    # -----------------------------------------------------------------
                    # -------------------------- range feature ------------------------
                    # -----------------------------------------------------------------
                    range_data, micro_range_data, range_angle_data = get_range_feature(adc_data, params)

                    # -----------------------------------------------------------------
                    # -------------------------- doppler feature ----------------------
                    # -----------------------------------------------------------------
                    micro_doppler_data, doppler_angle_data = get_doppler_feature(adc_data, params)

                    sci.savemat(img_dir + filename + '.mat',
                                {'range_data': range_data, 'micro_range_data': micro_range_data,
                                 'range_angle_data': range_angle_data, 'micro_doppler_data': micro_doppler_data,
                                 'doppler_angle_data': doppler_angle_data})

                    # -----------------------------------------------------------------
                    # -------------------------- angle feature ------------------------
                    # -----------------------------------------------------------------
                    # plot range-angle per frame

                    # plot range-angle per frame
