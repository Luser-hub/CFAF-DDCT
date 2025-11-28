import os

import pandas as pd
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import filtfilt,butter


def bandpass(data, low,high,sfreq):
    a,b= butter(6, [low,high],'bandpass',fs=sfreq)

    n_trials=data.shape[0]
    filtered_data=np.zeros(data.shape)
    for i in range(n_trials):
        filtered_data[i]=filtfilt(a,b,data[i],axis=1)


    return filtered_data


def read_single_limb_3_subj_data(sub_num1,start_num1):

    output_data = []
    output_labels = []
    for subj in range(start_num1,sub_num1+1):

        base_dir = ''

        subj_data_filepath = os.path.join(base_dir, f'sub{subj:02d}data.mat')
        subj_label_filepath = os.path.join(base_dir, f'sub{subj:02d}label.mat')
        data = sio.loadmat(subj_data_filepath)['data']
        label = sio.loadmat(subj_label_filepath)['label']
        data = data.transpose(2, 1, 0)
        traindata = []

        scaler = StandardScaler()
        for i in range(data.shape[0]):
            traindata.append(scaler.fit_transform(data[i][:,:750]))

        traindata = np.array(traindata)


        index = np.arange(traindata.shape[0])
        np.random.shuffle(index)
        traindata = traindata[index]
        label = label[:,index]


        output_data.append(traindata)
        output_labels.append(label)

    output_data= np.array(output_data)
    output_labels= np.array(output_labels)

    for i in range(len(output_data)):
        output_data[i]=bandpass(output_data[i],8,30,250)

    return output_data,output_labels

def loadData(i,sub_num1,start_num1):
    all_data, all_labels = read_single_limb_3_subj_data(sub_num1,start_num1)

    datas = []
    labels = []
    evaldata = []
    evallabel = []
    for j in range(sub_num1-start_num1+1):
        if j == i:
            trainX = all_data[j]
            label = all_labels[j]
            evaldata.append(trainX)
            evallabel.append(label)
            continue
        trainX = all_data[j]
        label = all_labels[j]
        datas.append(trainX)
        labels.append(label)
    datas = np.asarray(datas)
    labels = np.asarray(labels)
    evaldata = np.asarray(evaldata)
    evallabel = np.asarray(evallabel)

    # datas = datas.reshape(-1, 62, 800)
    datas = datas.reshape(-1, 60, 750)
    labels = labels.ravel()
    # evaldata = evaldata.reshape(-1, 62, 800)
    evaldata = evaldata.reshape(-1, 60, 750)
    evallabel = evallabel.ravel()

    OHlabels = pd.get_dummies(labels)
    OHlabels = np.array(OHlabels)
    OHevallabel = pd.get_dummies(evallabel)
    OHevallabel = np.array(OHevallabel)

    return datas, OHlabels,evaldata,OHevallabel

def loadSourceData(sub_num1,start_num1):
    all_data, all_labels = read_single_limb_3_subj_data(sub_num1,start_num1)

    datas = []
    labels = []
    evaldata = []
    evallabel = []
    for j in range(sub_num1-start_num1+1):

        trainX = all_data[j]
        label = all_labels[j]
        datas.append(trainX)
        labels.append(label)
    datas = np.asarray(datas)
    labels = np.asarray(labels)

    datas = datas.reshape(-1, 60, 750)
    labels = labels.ravel()

    OHlabels = pd.get_dummies(labels)
    OHlabels = np.array(OHlabels)


    return datas, OHlabels

def loadTargetData(i):
    all_data, all_labels = read_single_limb_3_subj_data(25,1)

    datas = []
    labels = []
    evaldata = []
    evallabel = []
    i=i-1
    for j in range(25):
        if j == i:
            trainX = all_data[j]
            label = all_labels[j]
            evaldata.append(trainX)
            evallabel.append(label)
            continue
        trainX = all_data[j]
        label = all_labels[j]
        datas.append(trainX)
        labels.append(label)
    datas = np.asarray(datas)
    labels = np.asarray(labels)
    evaldata = np.asarray(evaldata)
    evallabel = np.asarray(evallabel)

    datas = datas.reshape(-1, 60, 750)
    labels = labels.ravel()
    evaldata = evaldata.reshape(-1, 60, 750)
    evallabel = evallabel.ravel()

    OHlabels = pd.get_dummies(labels)
    OHlabels = np.array(OHlabels)
    OHevallabel = pd.get_dummies(evallabel)
    OHevallabel = np.array(OHevallabel)

    return evaldata,OHevallabel

