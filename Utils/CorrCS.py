import numpy as np
from tqdm import tqdm


def channel_weight_calculation(data, sfreq,window_size, step_size,voting,thresh=None):
    # calculate the channel weight for each channel based on the windows

    # based on sample frequency, window size and step size, calculate the number of windows
    max_window=data.shape[2]

    # generate the windows
    windows = []
    start_point = 0
    end_point = start_point*sfreq + window_size*sfreq
    while(end_point<= max_window):
        windows.append((start_point, end_point))
        start_point += step_size*sfreq
        end_point = start_point + window_size*sfreq
    windows= np.array(windows).astype(int)

    # for each window, calculate all trials correlation matrix
    channel_weight = []

    # first_trial_channel_weight = []

    for i in range(windows.shape[0]):
        window_data = data[:,:,windows[i,0]:windows[i,1]]
        local_window_result = []
        # calculate the correlation matrix for all trials and add them to the local_window_result
        for j in tqdm(range(window_data.shape[0]),desc='Calculating channel weight of window '+str(i+1)+'...'):
            corr_matrix = np.corrcoef(window_data[j])

            # sum by row
            channel_mean_weight = np.sum(corr_matrix,axis=1)/window_data.shape[1]

            # soft voting or hard voting
            if voting=='soft':
                local_window_result.append(channel_mean_weight)
            elif voting=='hard_mean':
                thresh = np.mean(channel_mean_weight)
                local_window_result.append(np.where(channel_mean_weight>thresh,1,0))
            elif voting=='hard_median':
                thresh = np.median(channel_mean_weight)
                local_window_result.append(np.where(channel_mean_weight>thresh,1,0))

        # first_trial_channel_weight.append(local_window_result[0])

        local_window_result = np.sum(np.array(local_window_result),axis=0)/window_data.shape[0]
        channel_weight.append(local_window_result)
    channel_weight = np.sum(np.array(channel_weight),axis=0)/windows.shape[0]
    channel_weight = channel_weight/np.sum(channel_weight)
    return channel_weight
