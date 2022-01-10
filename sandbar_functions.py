from scipy.signal import find_peaks
import numpy as np
import pandas as pd


# -----------------------------------------------------------------
#
# IDENTIFY SAND BAR POINTS
#
# -----------------------------------------------------------------
def identify_sandbar_pts(average_img, orientation, window, prominence):
    """
    Returns two lists with all sandbar points
    :param orientation: Waves direction
    :param average_img: cumulative breaking pixels map
    :return: x_point and y_point
    """
    x_point, y_point = list(), list()
    for i in range(len(average_img[0, :])):
        if orientation == 'vertical':
            row = average_img[:, i]
        else:
            row = average_img[i, :]
        peaks, _ = find_peaks(smooth_hamming(row, window), prominence=prominence)
        for peak in peaks:
            x_point.append(i)
            y_point.append(peak)

    return x_point, y_point


# -----------------------------------------------------------------
#
# FILTER TO
#
# -----------------------------------------------------------------
def smooth_hamming(ts, window):
    """
    Hamming smooth function using pandas
    :param ts: 1D np array
    :param window: number of points for hamming filter
    :return: smooth_ts
    """
    df = pd.DataFrame(columns=['ts'])
    df.ts = ts
    df['smooth_ts'] = df.iloc[:, 0].rolling(window,
                                            min_periods=1,
                                            win_type='hamming',
                                            center=True).mean()
    smooth_ts = np.squeeze(np.array(df.smooth_ts))
    return smooth_ts