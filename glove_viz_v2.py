import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as p3


# update 8/3/2018:
# new visualization methods from glove_tcp


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
NUM_SENSORS = 17


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


def load_cal(ID, trial, hand):
    cal_rotm = np.zeros([3, 3, NUM_SENSORS])
    calpath = 'D:\python_program\HandMotionPrimitive\data\withlabel_v2\ID'+str(ID)+'\\feature\\'+hand+'_IMU_'+str(trial)+'_calrotm.csv'
    caldata = np.genfromtxt(calpath, delimiter=",")
    for i in range(NUM_SENSORS):
        cal_rotm[:, :, i] = caldata[i*3:i*3+3, :]
    return cal_rotm


def plot_posture(quaternions, ID, trial, hand, label):

    if hand == 'R':
        IMUIDs = np.array(
            [6, 7, 8, 3, 4, 5, 12, 13, 14, 9, 10, 11, 0, 1,
             2])  # thumb root to tip -> index root to tip -> ... {1,2,3,...}
    else:
        IMUIDs = np.array(
            [6, 7, 8, 0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4,
             5])  # thumb root to tip -> index root to tip -> ... {1,2,3,...}

    data = np.reshape(quaternions, (1, len(quaternions)))
    thumb = SGloveFingers(3, IMUIDs, 0)
    index = SGloveFingers(3, IMUIDs, 3)
    middle = SGloveFingers(3, IMUIDs, 6)
    engage = SGloveFingers(3, IMUIDs, 9)
    pinky = SGloveFingers(3, IMUIDs, 12)

    calrotm = load_cal(ID, trial, hand)

    origin = quat2rotm(quaternions[REFERENCE_FRAME_ID * 4:REFERENCE_FRAME_ID * 4 + 4])
    numParm = 4
    vPalm = np.zeros([numParm, 3])
    vFingerBase = np.zeros([NUM_FINGERS, 3])

    if hand == 'R':
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
    vFingerBase[2, :] = vPalm[1, :] + 0.3 * (vPalm[2, :] - vPalm[1, :])
    vFingerBase[3, :] = vPalm[1, :] + 0.6 * (vPalm[2, :] - vPalm[1, :])
    vFingerBase[4, :] = vPalm[2, :]

    vecSeg = np.array([-4, 0, 0])
    thumb.update_infoAllSeg(quaternions, vecSeg, vFingerBase[0, :], calrotm, origin)
    index.update_infoAllSeg(quaternions, vecSeg, vFingerBase[1, :], calrotm, origin)
    middle.update_infoAllSeg(quaternions, vecSeg, vFingerBase[2, :], calrotm, origin)
    engage.update_infoAllSeg(quaternions, vecSeg, vFingerBase[3, :], calrotm, origin)
    pinky.update_infoAllSeg(quaternions, vecSeg, vFingerBase[4, :], calrotm, origin)

    pPalm = vPalm
    pFingerBase = vFingerBase

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

    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')

    if hand == 'R':
        ax.view_init(elev=45, azim=220)
    else:
        ax.view_init(elev=45, azim=140)

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

    coord_palm = np.reshape(strip_palm, (1, 15))
    coord_thumb = np.reshape(list_fingerSeg[0:4, :], (1, 12))
    coord_index = np.reshape(list_fingerSeg[4:8, :], (1, 12))
    coord_middle = np.reshape(list_fingerSeg[8:12, :], (1, 12))
    coord_engage = np.reshape(list_fingerSeg[12:16, :], (1, 12))
    coord_pinky = np.reshape(list_fingerSeg[16:20, :], (1, 12))

    x_id = [0, 3, 6, 9]
    y_id = [1, 4, 7, 10]
    z_id = [2, 5, 8, 11]

    thumb_line, = ax.plot(coord_thumb[0, x_id], coord_thumb[0, y_id], coord_thumb[0, z_id], 'co-', fillstyle='none',
                          ms=10, label='thumb')
    index_line, = ax.plot(coord_index[0, x_id], coord_index[0, y_id], coord_index[0, z_id], 'bo-', fillstyle='none',
                          ms=10, label='index')
    middle_line, = ax.plot(coord_middle[0, x_id], coord_middle[0, y_id], coord_middle[0, z_id], 'ro-', fillstyle='none',
                           ms=10, label='middle')
    engage_line, = ax.plot(coord_engage[0, x_id], coord_engage[0, y_id], coord_engage[0, z_id], 'go-', fillstyle='none',
                           ms=10, label='ring')
    pinky_line, = ax.plot(coord_pinky[0, x_id], coord_pinky[0, y_id], coord_pinky[0, z_id], 'yo-', fillstyle='none',
                          ms=10, label='pinkie')

    x_palm = [0, 3, 6, 9, 0]
    y_palm = [1, 4, 7, 10, 1]
    z_palm = [2, 5, 8, 11, 2]
    palm_line, = ax.plot(coord_palm[0, x_palm], coord_palm[0, y_palm], coord_palm[0, z_palm], 'ko-', fillstyle='none',
                         ms=10, label='palm')

    plt.legend()
    title = 'Cluster center # ' + str(label)
    plt.title(title)
    plt.show()

