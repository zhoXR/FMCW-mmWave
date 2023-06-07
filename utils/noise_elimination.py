import numpy as np


def noise_elimination(azimuth_fft_abs, numADCSamples, numLoopsPerFrame, numAngleBins):
    doppler_intensity = np.sum(np.sum(azimuth_fft_abs, 0), 1)
    doppler_ave = doppler_intensity.mean(0)
    cul = doppler_intensity > doppler_ave
    ans = np.zeros((numAngleBins, numADCSamples))
    for d in range(numLoopsPerFrame):
        if (cul[d]):
            ans += np.squeeze(azimuth_fft_abs[:, d, :])
    return ans
