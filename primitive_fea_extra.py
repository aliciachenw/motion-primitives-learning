import numpy as np
import os
import csv
import tsfeature.feature_core as feature_core

num_sensors = 17
ID = 15456
hand = 'R'

if hand == 'L':
    handpath = 'left'
elif hand == 'R':
    handpath = 'right'

feapath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID'+str(ID)+'\\'+handpath+'\primitive\\tf_fea_angle.csv'
feafile = open(feapath, "w", newline='')
feawriter = csv.writer(feafile)

refpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID'+str(ID)+'\\'+handpath+'\primitive\\ngram_window\pri_ngram_window.csv'
refdata = np.genfromtxt(refpath, delimiter=',')
refinfo = refdata[:, 0:5]
# [obj, intent, trial, begin_stamp, end_stamp]

for i in range(refinfo.shape[0]):
    trial = int(refinfo[i, 2])
    stp = int(refinfo[i, 3])
    endp = int(refinfo[i, 4])
    trialfile = 'D:\python_program\HandMotionPrimitive\data\withlabel_v3\ID' + str(ID) + '\\feature\\' + hand + '_IMU_' + str(
        trial) + '_fea.csv'
    trial_data = np.genfromtxt(trialfile, delimiter=',')
    pri_data = trial_data[stp:endp, 2:22]  # angle: 2:22, raw quaternion: 2:2+4*num_sensors
    feature = []
    for j in range(pri_data.shape[1]):
        cha_feature = feature_core.sequence_feature(pri_data[:,j], 40, 20)
        fc = np.hstack((range(3), range(4, 7), range(14, 17)))
        cha_feature = cha_feature[:, fc]
        cha_feature = np.reshape(cha_feature, cha_feature.shape[0] * cha_feature.shape[1])
        feature = np.hstack((feature, cha_feature))
    feature = np.hstack((refinfo[i, :], feature))
    feawriter.writerow(feature)
