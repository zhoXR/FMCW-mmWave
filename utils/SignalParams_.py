import mmwave.dsp as dsp
import numpy as np


class SignalParams:

    def __init__(self):  # self必须作为函数的第一个参数
        self.c = 3e8  # Speed of light (m/s)
        self.numFrames = 10
        self.numADCSamples = 80
        self.numTxAntennas = 2
        self.numRxAntennas = 4
        self.numLoopsPerFrame = 255
        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame

        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame
        self.numAngleBins = 64
        self.framePeriod = 0.04
        self.sampleRate = 2499
        self.slope = 100
        self.start_freq = 77
        self.ramp_end_time = 39.74
        self.idle_time_const = 7

        self.range_resolution, self.bandwidth = dsp.range_resolution(self.numADCSamples,
                                                                     dig_out_sample_rate=self.sampleRate,
                                                                     freq_slope_const=self.slope)
        self.doppler_resolution = dsp.doppler_resolution(self.bandwidth, start_freq_const=self.start_freq,
                                                         ramp_end_time=self.ramp_end_time,
                                                         idle_time_const=self.idle_time_const,
                                                         num_loops_per_frame=self.numLoopsPerFrame,
                                                         num_tx_antennas=self.numTxAntennas)
        self.times = np.arange(self.numFrames) * self.framePeriod
        self.ranges = np.arange(
            self.numADCSamples) * self.range_resolution  # Apply the range resolution factor to the range indices
        self.velocities = (np.arange(self.numLoopsPerFrame) - (
                self.numLoopsPerFrame // 2)) * self.doppler_resolution  # Apply the velocity resolution factor to the velocity indices
