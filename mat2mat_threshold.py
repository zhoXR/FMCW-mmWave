import os
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from utils.SignalParams import SignalParams

params = SignalParams()
# -----------------------------------------------------------------
# -------------------------- 导入数据 ------------------------------
# -----------------------------------------------------------------
dirname = 'G:/B科研/data_code/mydata/20211214/angle2_range_doppler_data/'
gestures = 10
persons = {'kx'}
conditions = [1, 2, 3, 6, 7]
counts = 5
mat_dir = 'G:/B科研/data_code/mydata/20211214/angle2_range_doppler_mat_threshold/'
for gesId in range(gestures):
    for person in persons:
        for conId in conditions:
            for timeId in range(counts):
                filename = str(gesId + 1).zfill(2) + '_' + person + '_' + str(timeId + 1).zfill(2) + '_' + str(
                    conId).zfill(2)
                filepath = dirname + filename + '.mat'
                if not os.path.exists(filepath):
                    print(filepath + ' is not exist!')
                else:
                    src_data = sci.loadmat(filepath)
                    # range_data, micro_range_data, range_angle_data, micro_doppler_data, doppler_angle_data
                    range_angle_data = src_data['range_angle_data']
                    doppler_angle_data = src_data['doppler_angle_data']
                    # range-time plot
                    data = range_angle_data.sum(axis=1).T
                    data_log2 = np.log2(data[0:40, :])
                    data_log10 = np.log10(data[0:40, :])
                    a, b, c = plt.hist(data_log10.ravel(), bins=11, fc='k', ec='k')
                    # plt.show()

                    index_max = np.argmax(a)
                    threshold = b[index_max + 2]
                    range_time_data = np.where(data_log10 > threshold, data_log10, np.min(data_log10))
                    range_time_data = range_time_data.reshape(1, -1)
                    range_time_data = minmax_scale(range_time_data, feature_range=(-1, 1), axis=1)
                    range_time_data = range_time_data.reshape(-1, params.numFrames)

                    # doppler-time plot
                    data = doppler_angle_data.sum(axis=1).T
                    data_log2 = np.log2(data[87:167, :])
                    data_log10 = np.log10(data[87:167, :])
                    a, b, c = plt.hist(data_log10.ravel(), bins=11, fc='k', ec='k')
                    # plt.show()

                    index_max = np.argmax(a)
                    threshold = b[index_max + 2]
                    doppler_time_data = np.where(data_log10 > threshold, data_log10, np.min(data_log10))
                    doppler_time_data = doppler_time_data.reshape(1, -1)
                    doppler_time_data = minmax_scale(doppler_time_data, feature_range=(-1, 1), axis=1)
                    doppler_time_data = doppler_time_data.reshape(-1, params.numFrames)

                    sci.savemat(mat_dir + filename + '.mat',
                                {'range_time_data': range_time_data, 'doppler_time_data': doppler_time_data})

                    print(filepath)
