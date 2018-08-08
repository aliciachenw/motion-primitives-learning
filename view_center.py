import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
from glove_viz_v2 import plot_posture
import csv
import numpy as np

num_sensors = 17

ID = 15456
hand = 'R'
feature = False

if hand == 'L':
    handpath = 'left'
elif hand == 'R':
    handpath = 'right'
if feature:
    feapath = 'feature'
else:
    feapath = 'raw'
csvpath = 'D:\\python_program\HandMotionPrimitive\Clustering\\results_v3\ID'+str(ID)+'\\'+handpath+'\\'+feapath+'\center.csv'
data = np.genfromtxt(csvpath, delimiter=',')
for i in range(data.shape[0]):
    trial = int(data[i, data.shape[1] - 2])
    time_stamp = int(data[i, data.shape[1] - 1])
    trialfile = 'D:\python_program\HandMotionPrimitive\data\withlabel_v2\ID' + str(ID) + '\\' + hand + '_IMU_' + str(
        trial) + '.csv'
    sensor_data = np.genfromtxt(trialfile, delimiter=",")
    cen_sensor_data = sensor_data[time_stamp, 2:2 + 4 * num_sensors]
    plot_posture(cen_sensor_data, ID, trial, hand, i)
