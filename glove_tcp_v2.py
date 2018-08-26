import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as p3
import matplotlib.animation as an
import time
import wand
import sys

class SGloveFingers:
    def __init__(self, _numSeg, _IMUIDs, startP):
        self.IMUIDs = _IMUIDs[startP:startP + _numSeg]
        self.IMURots = np.zeros([_numSeg, 3, 3])
        self.segPoints = np.zeros([_numSeg, 3])
        self.numSegment = _numSeg

    def update_info(self, SegIdx, quaternions, nextPos, prevPos, Calm, origin):
        qIdx = 4 * self.IMUIDs[SegIdx]
        self.IMURots[SegIdx, :, :] = quat2rotm(quaternions[qIdx:qIdx + 4])
        rotm = np.matmul(self.IMURots[SegIdx, :, :], Calm[:, :, self.IMUIDs[SegIdx]])
        rotm = np.matmul(rotm, np.linalg.inv(origin))
        self.segPoints[SegIdx, :] = prevPos + np.matmul(rotm, nextPos)
        return self.segPoints[SegIdx, :]

    def update_infoAllSeg(self, quaternions, segPos, rootPos, Calm, origin):
        for i in range(self.numSegment):
            rootPos = self.update_info(i, quaternions, segPos, rootPos, Calm, origin)


DATA_LEN = 4
ROUND_POINT = 1e-4
REFERENCE_FRAME_ID = 15
SECOND_PALM_ID = 16
NUM_FINGERS = 5
ID = 15456
hand = 'left'
trial = 8


def quat2rotm(q):
    n = 1.0 / np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    q[0] = q[0] * n
    q[1] = q[1] * n
    q[2] = q[2] * n
    q[3] = q[3] * n
    rot3M = np.zeros((3, 3))
    rot3M[0, 0] = 1.0 - 2.0 * q[2] * q[2] - 2.0 * q[3] * q[3]
    rot3M[0, 1] = 2.0 * q[1] * q[2] - 2.0 * q[3] * q[0]
    rot3M[0, 2] = 2.0 * q[1] * q[3] + 2.0 * q[2] * q[0]
    rot3M[1, 0] = 2.0 * q[1] * q[2] + 2.0 * q[3] * q[0]
    rot3M[1, 1] = 1.0 - 2.0 * q[1] * q[1] - 2.0 * q[3] * q[3]
    rot3M[1, 2] = 2.0 * q[2] * q[3] - 2.0 * q[1] * q[0]
    rot3M[2, 0] = 2.0 * q[1] * q[3] - 2.0 * q[2] * q[0]
    rot3M[2, 1] = 2.0 * q[2] * q[3] + 2.0 * q[1] * q[0]
    rot3M[2, 2] = 1.0 - 2.0 * q[1] * q[1] - 2.0 * q[2] * q[2]
    for i in range(3):
        n = 1.0 / np.sqrt(rot3M[i, 0] ** 2 + rot3M[i, 1] ** 2 + rot3M[i, 2] ** 2)
        rot3M[i, 0] = rot3M[i, 0] * n
        rot3M[i, 1] = rot3M[i, 1] * n
        rot3M[i, 2] = rot3M[i, 2] * n
    # rot3M = np.swapaxes(rot3M, 0, 1)
    return rot3M


if hand == 'right':
    IMUIDs = np.array(
        [6, 7, 8, 3, 4, 5, 12, 13, 14, 9, 10, 11, 0, 1, 2])  # thumb root to tip -> index root to tip -> ... {1,2,3,...}
else:
    IMUIDs = np.array(
        [6, 7, 8, 0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5])  # thumb root to tip -> index root to tip -> ... {1,2,3,...}


thumb = SGloveFingers(3, IMUIDs, 0)
index = SGloveFingers(3, IMUIDs, 3)
middle = SGloveFingers(3, IMUIDs, 6)
engage = SGloveFingers(3, IMUIDs, 9)
pinky = SGloveFingers(3, IMUIDs, 12)


def calibrate_glove(cal_quat):
    theta = np.array([131.65/180*np.pi, 101.61/180*np.pi, 86.19/180*np.pi, 77.31/180*np.pi, 54.52/180*np.pi])
    if hand == 'left':
        std_pose = np.array([[np.cos(np.pi / 2 - theta[1]), np.sin(np.pi / 2 - theta[1]), 0],
                             [np.cos(np.pi / 2 - theta[1]), np.sin(np.pi / 2 - theta[1]), 0],
                             [np.cos(np.pi / 2 - theta[1]), np.sin(np.pi / 2 - theta[1]), 0],
                             [np.cos(np.pi / 2 - theta[4]), np.sin(np.pi / 2 - theta[4]), 0],
                             [np.cos(np.pi / 2 - theta[4]), np.sin(np.pi / 2 - theta[4]), 0],
                             [np.cos(np.pi / 2 - theta[4]), np.sin(np.pi / 2 - theta[4]), 0],
                             [np.cos(np.pi / 2 - theta[0]), np.sin(np.pi / 2 - theta[0]), 0],
                             [np.cos(np.pi / 2 - theta[0]), np.sin(np.pi / 2 - theta[0]), 0],
                             [np.cos(np.pi / 2 - theta[0]), np.sin(np.pi / 2 - theta[0]), 0],
                             [np.cos(np.pi / 2 - theta[2]), np.sin(np.pi / 2 - theta[2]), 0],
                             [np.cos(np.pi / 2 - theta[2]), np.sin(np.pi / 2 - theta[2]), 0],
                             [np.cos(np.pi / 2 - theta[2]), np.sin(np.pi / 2 - theta[2]), 0],
                             [np.cos(np.pi / 2 - theta[3]), np.sin(np.pi / 2 - theta[3]), 0],
                             [np.cos(np.pi / 2 - theta[3]), np.sin(np.pi / 2 - theta[3]), 0],
                             [np.cos(np.pi / 2 - theta[3]), np.sin(np.pi / 2 - theta[3]), 0],
                             [np.cos(0), np.sin(0), 0],
                             [np.cos(0), np.sin(0), 0],
                             ])
    if hand == 'right':
        std_pose = np.array([[np.cos(theta[4] - np.pi / 2), np.sin(theta[4] - np.pi / 2), 0],
                             [np.cos(theta[4] - np.pi / 2), np.sin(theta[4] - np.pi / 2), 0],
                             [np.cos(theta[4] - np.pi / 2), np.sin(theta[4] - np.pi / 2), 0],
                             [np.cos(theta[1] - np.pi / 2), np.sin(theta[1] - np.pi / 2), 0],
                             [np.cos(theta[1] - np.pi / 2), np.sin(theta[1] - np.pi / 2), 0],
                             [np.cos(theta[1] - np.pi / 2), np.sin(theta[1] - np.pi / 2), 0],
                             [np.cos(theta[0] - np.pi / 2), np.sin(theta[0] - np.pi / 2), 0],
                             [np.cos(theta[0] - np.pi / 2), np.sin(theta[0] - np.pi / 2), 0],
                             [np.cos(theta[0] - np.pi / 2), np.sin(theta[0] - np.pi / 2), 0],
                             [np.cos(theta[3] - np.pi / 2), np.sin(theta[3] - np.pi / 2), 0],
                             [np.cos(theta[3] - np.pi / 2), np.sin(theta[3] - np.pi / 2), 0],
                             [np.cos(theta[3] - np.pi / 2), np.sin(theta[3] - np.pi / 2), 0],
                             [np.cos(theta[2] - np.pi / 2), np.sin(theta[2] - np.pi / 2), 0],
                             [np.cos(theta[2] - np.pi / 2), np.sin(theta[2] - np.pi / 2), 0],
                             [np.cos(theta[2] - np.pi / 2), np.sin(theta[2] - np.pi / 2), 0],
                             [np.cos(0), np.sin(0), 0],
                             [np.cos(0), np.sin(0), 0],
                             ])
    cal_rotm = np.zeros([3, 3, 17])
    # TODO: MODIFY THE CALIBRATION!
    for i in range(17):
        framem = quat2rotm(cal_quat[REFERENCE_FRAME_ID*4:REFERENCE_FRAME_ID*4+4])
        rotm = quat2rotm(cal_quat[i*4:i*4+4])
        stdm = np.array([[std_pose[i, 0], -std_pose[i, 1], 0],
                         [std_pose[i, 1], std_pose[i, 0], 0],
                         [0, 0, 1]
                         ])
        cal_rotm[:, :, i] = np.matmul(stdm, framem)
        cal_rotm[:, :, i] = np.matmul(np.linalg.inv(rotm), cal_rotm[:, :, i])
        for j in range(3):
            n = 1.0 / np.sqrt(cal_rotm[0, j, i] ** 2 + cal_rotm[1, j, i] ** 2 + cal_rotm[2, j, i] ** 2)
            cal_rotm[0, j, i] = cal_rotm[0, j, i] * n
            cal_rotm[1, j, i] = cal_rotm[1, j, i] * n
            cal_rotm[2, j, i] = cal_rotm[2, j, i] * n
    return cal_rotm


def update_glove(quaternions):
    origin = quat2rotm(quaternions[REFERENCE_FRAME_ID * 4:REFERENCE_FRAME_ID * 4 + 4])
    numParm = 4
    vPalm = np.zeros([numParm, 3])
    vFingerBase = np.zeros([NUM_FINGERS, 3])

    if hand == 'right':
        vPalm[0, :] = [6, -4.5, 1.25]
        vPalm[1, :] = [-6, -4, 0.75]
        vPalm[2, :] = [-6, 4.5, 0.75]
        vPalm[3, :] = [6, 4.5, 0.75]
    else:
        vPalm[0, :] = [6, 4.5, 1.25]
        vPalm[1, :] = [-6, 4, 0.75]
        vPalm[2, :] = [-6, -4.5, 0.75]
        vPalm[3, :] = [6, -4.5, 0.75]

    vFingerBase[0, :] = vPalm[0, :]
    vFingerBase[1, :] = vPalm[1, :]
    vFingerBase[2, :] = vPalm[1, :] + 0.3 * (vPalm[2, :]-vPalm[1, :])
    vFingerBase[3, :] = vPalm[1, :] + 0.6 * (vPalm[2, :]-vPalm[1, :])
    vFingerBase[4, :] = vPalm[2, :]

    vecSeg = np.array([-4, 0, 0])
    thumb.update_infoAllSeg(quaternions, vecSeg, vFingerBase[0, :], calrotm, origin)
    index.update_infoAllSeg(quaternions, vecSeg, vFingerBase[1, :], calrotm, origin)
    middle.update_infoAllSeg(quaternions, vecSeg, vFingerBase[2, :], calrotm, origin)
    engage.update_infoAllSeg(quaternions, vecSeg, vFingerBase[3, :], calrotm, origin)
    pinky.update_infoAllSeg(quaternions, vecSeg, vFingerBase[4, :], calrotm, origin)

    pPalm = vPalm
    pFingerBase = vFingerBase

    point_nodes = pPalm
    strip_palm = pPalm
    strip_palm = np.vstack((strip_palm, pPalm[0, :]))

    list_fingerSeg = pFingerBase[0, :]
    list_fingerSeg = np.vstack((list_fingerSeg, thumb.segPoints[0:thumb.numSegment, :]))
    list_fingerSeg = np.vstack((list_fingerSeg, pFingerBase[1, :]))
    list_fingerSeg = np.vstack((list_fingerSeg, index.segPoints[0:index.numSegment, :]))
    list_fingerSeg = np.vstack((list_fingerSeg, pFingerBase[2, :]))
    list_fingerSeg = np.vstack((list_fingerSeg, middle.segPoints[0:middle.numSegment, :]))
    list_fingerSeg = np.vstack((list_fingerSeg, pFingerBase[3, :]))
    list_fingerSeg = np.vstack((list_fingerSeg, engage.segPoints[0:engage.numSegment, :]))
    list_fingerSeg = np.vstack((list_fingerSeg, pFingerBase[4, :]))
    list_fingerSeg = np.vstack((list_fingerSeg, pinky.segPoints[0:pinky.numSegment, :]))

    return point_nodes, strip_palm, list_fingerSeg


def animate_glove(num):
    x_id = [0, 3, 6, 9]
    y_id = [1, 4, 7, 10]
    z_id = [2, 5, 8, 11]
    x_palm = [0, 3, 6, 9, 0]
    y_palm = [1, 4, 7, 10, 1]
    z_palm = [2, 5, 8, 11, 2]

    if num < num_frames:
        thumb_line.set_xdata(coord_thumb[num, x_id])
        thumb_line.set_ydata(coord_thumb[num, y_id])
        thumb_line.set_3d_properties(coord_thumb[num, z_id])

        index_line.set_xdata(coord_index[num, x_id])
        index_line.set_ydata(coord_index[num, y_id])
        index_line.set_3d_properties(coord_index[num, z_id])

        middle_line.set_xdata(coord_middle[num, x_id])
        middle_line.set_ydata(coord_middle[num, y_id])
        middle_line.set_3d_properties(coord_middle[num, z_id])

        engage_line.set_xdata(coord_engage[num, x_id])
        engage_line.set_ydata(coord_engage[num, y_id])
        engage_line.set_3d_properties(coord_engage[num, z_id])

        pinky_line.set_xdata(coord_pinky[num, x_id])
        pinky_line.set_ydata(coord_pinky[num, y_id])
        pinky_line.set_3d_properties(coord_pinky[num, z_id])

        palm_line.set_xdata(coord_palm[num, x_palm])
        palm_line.set_ydata(coord_palm[num, y_palm])
        palm_line.set_3d_properties(coord_palm[num, z_palm])
    else:
        quit()

    return thumb_line, index_line, middle_line, engage_line, pinky_line, palm_line


if __name__ == "__main__":
    t0 = time.clock()
    plt.rcParams['animation.convert_path'] = 'C:\ImageMagick\magick.exe'
    if hand == 'right':
        # data = np.genfromtxt('sample_data_R.csv', delimiter=",")
        datafile = 'D:\python_program\HandMotionPrimitive\data\withlabel_v3\ID'+str(ID)+'\R_IMU_'+str(trial)+'.csv'
    if hand == 'left':
        # data = np.genfromtxt('sample_data_L.csv', delimiter=",")
        datafile = 'D:\python_program\HandMotionPrimitive\data\withlabel_v3\ID'+str(ID)+'\L_IMU_' + str(trial) + '.csv'

    data = np.genfromtxt(datafile, delimiter=",")
    data = data[:, 2:70]

    if data.ndim == 1:  # this is the case when we use 'ground_quats_R or L', where we only have one row
        data = data.reshape(1, data.shape[0])

    calrotm = calibrate_glove(data[0, :])

    num_frames = data.shape[0]
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')

    if hand == 'right':
        ax.view_init(elev=45, azim=220)
    else:
        ax.view_init(elev=45, azim=140)

    point_nodes = []
    strip_palm = []
    list_fingerSeg = []

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    lim = 20
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    plt.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    coord_palm = np.zeros((data.shape[0], 15))
    coord_thumb = np.zeros((data.shape[0], 12))
    coord_index = np.zeros((data.shape[0], 12))
    coord_middle = np.zeros((data.shape[0], 12))
    coord_engage = np.zeros((data.shape[0], 12))
    coord_pinky = np.zeros((data.shape[0], 12))

    for i in range(data.shape[0]):
        point_nodes, strip_palm, list_fingerSeg = update_glove(data[i, :])
        coord_palm[i, :] = np.reshape(strip_palm, 15)
        coord_thumb[i, :] = np.reshape(list_fingerSeg[0:4, :], 12)
        coord_index[i, :] = np.reshape(list_fingerSeg[4:8, :], 12)
        coord_middle[i, :] = np.reshape(list_fingerSeg[8:12, :], 12)
        coord_engage[i, :] = np.reshape(list_fingerSeg[12:16, :], 12)
        coord_pinky[i, :] = np.reshape(list_fingerSeg[16:20, :], 12)

    x_id = [0, 3, 6, 9]
    y_id = [1, 4, 7, 10]
    z_id = [2, 5, 8, 11]

    thumb_line, = ax.plot(coord_thumb[0, x_id], coord_thumb[0, y_id], coord_thumb[0, z_id], 'co-', fillstyle='none', ms=10, label='thumb')
    index_line, = ax.plot(coord_index[0, x_id], coord_index[0, y_id], coord_index[0, z_id], 'bo-', fillstyle='none', ms=10, label='index')
    middle_line, = ax.plot(coord_middle[0, x_id], coord_middle[0, y_id], coord_middle[0, z_id], 'ro-', fillstyle='none', ms=10, label='middle')
    engage_line, = ax.plot(coord_engage[0, x_id], coord_engage[0, y_id], coord_engage[0, z_id], 'go-', fillstyle='none', ms=10, label='ring')
    pinky_line, = ax.plot(coord_pinky[0, x_id], coord_pinky[0, y_id], coord_pinky[0, z_id], 'yo-', fillstyle='none', ms=10, label='pinkie')

    x_palm = [0, 3, 6, 9, 0]
    y_palm = [1, 4, 7, 10, 1]
    z_palm = [2, 5, 8, 11, 2]
    palm_line, = ax.plot(coord_palm[0, x_palm], coord_palm[0, y_palm], coord_palm[0, z_palm], 'ko-', fillstyle='none', ms=10,label='palm')


    print(time.clock() - t0)
    plt.legend()
    ani = an.FuncAnimation(fig, animate_glove, interval=100, blit=False)
    plt.show()
