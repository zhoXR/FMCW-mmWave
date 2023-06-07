import mmwave.dsp as dsp
from mmwave.dsp.utils import Window
import numpy as np
from scipy import signal


def get_doppler_feature(adc_data, params):
    sos1 = signal.iirfilter(25, [30, 70], rs=80, btype='band',
                            analog=False, ftype='cheby2', fs=160,
                            output='sos')
    sos2 = signal.iirfilter(25, [5, 30], rs=80, btype='band',
                            analog=False, ftype='cheby2', fs=255,
                            output='sos')

    micro_doppler_data = np.zeros(
        (params.numFrames, params.numTxAntennas * params.numRxAntennas, params.numLoopsPerFrame, params.numADCSamples),
        dtype=np.float64)
    aoa_input_data = np.zeros(
        (params.numFrames, params.numTxAntennas * params.numRxAntennas, params.numLoopsPerFrame, params.numADCSamples),
        dtype=np.complex_)
    doppler_angle_res = np.zeros((params.numFrames, params.numAngleBins, params.numLoopsPerFrame), dtype=np.float64)

    for i, frame in enumerate(adc_data):
        frame_allVirtualAntenna = dsp.separate_tx(frame, params.numTxAntennas, vx_axis=1, axis=0)
        for virantennaIdx in range(params.numTxAntennas * params.numRxAntennas):

            frame_singleAntenna = frame_allVirtualAntenna[:, virantennaIdx, :]

            frame_singleAntenna_filtered = signal.sosfiltfilt(sos1, frame_singleAntenna, axis=1, padlen=int(
                params.numADCSamples / 3))  # sosfiltfilt default axis = -1
            frame_singleAntenna = np.expand_dims(frame_singleAntenna_filtered, 1)

            # Range Processing
            range_res = dsp.range_processing(frame_singleAntenna, window_type_1d=Window.HANNING)

            # Doppler Processing
            det_matrix, aoa_input = dsp.doppler_processing(range_res, num_tx_antennas=1, clutter_removal_enabled=False,
                                                           window_type_2d=Window.HANNING, interleaved=True,
                                                           accumulate=True)

            # --- Show output
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            micro_doppler_data[i, virantennaIdx, :, :] = det_matrix_vis.T
            aoa_input_data[i, virantennaIdx, :, :] = np.squeeze(aoa_input).T

        # (4) Angle Processing
        aoa_input_data_shift = np.fft.fftshift(np.squeeze(aoa_input_data[i, 0:params.numRxAntennas * 2, :, :]), axes=1)
        azimuth_fft = np.fft.fft(aoa_input_data_shift, params.numAngleBins, axis=0)
        azimuth_fft = np.fft.fftshift(azimuth_fft, axes=0)
        azimuth_fft_abs = np.abs(azimuth_fft)

        doppler_angle = np.sum(azimuth_fft_abs, axis=2)
        doppler_angle_res[i, :, :] = doppler_angle


    return micro_doppler_data, doppler_angle_res
