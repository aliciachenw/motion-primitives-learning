import numpy as np
import os
import time
import csv


hand = 'L'
ID = 80237
trial = 6


class SGloveFingers:
    def __init__(self, _numSeg, _IMUIDs, startP):
        self.IMUIDs = _IMUIDs[startP:startP + _numSeg]
        self.IMURots = np.zeros([_numSeg, 3, 3])
        self.orientation = np.zeros([_numSeg, 3])
        self.numSegment = _numSeg

    def update_info(self, SegIdx, quaternions, Pos, Calm, origin):
        qIdx = 4 * self.IMUIDs[SegIdx]
        self.IMURots[SegIdx, :, :] = quat2rotm(quaternions[qIdx:qIdx + 4])
        rotm = np.matmul(self.IMURots[SegIdx, :, :], Calm[:, :, self.IMUIDs[SegIdx]])
        rotm = np.matmul(rotm, np.linalg.inv(origin))
        self.orientation[SegIdx, :] = np.matmul(rotm, Pos)

    def update_infoAllSeg(self, quaternions, segPos, Calm, origin):
        for i in range(self.numSegment):
            self.update_info(i, quaternions, segPos, Calm, origin)


DATA_LEN = 4
REFERENCE_FRAME_ID = 15
SECOND_PALM_ID = 16
NUM_FINGERS = 5
NUM_FEATURES_FINGER = 3
NUM_FEATURES = NUM_FEATURES_FINGER*NUM_FINGERS + NUM_FINGERS - 1


def get_norm(vec):
    return np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)


def get_angle(vec1, vec2):
    angle = (vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) / (get_norm(vec1) * get_norm(vec2))
    if angle >= 1.0:
        angle = 0.0
    else:
        angle = np.arccos(angle)
    angle = angle / np.pi
    return angle


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


if hand == 'R':
    IMUIDs = np.array(
        [6, 7, 8, 3, 4, 5, 12, 13, 14, 9, 10, 11, 0, 1, 2])  # thumb root to tip -> index root to tip -> ... {1,2,3,...}
if hand == 'L':
    IMUIDs = np.array(
        [6, 7, 8, 0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5])  # thumb root to tip -> index root to tip -> ... {1,2,3,...}

thumb = SGloveFingers(3, IMUIDs, 0)
index = SGloveFingers(3, IMUIDs, 3)
middle = SGloveFingers(3, IMUIDs, 6)
engage = SGloveFingers(3, IMUIDs, 9)
pinky = SGloveFingers(3, IMUIDs, 12)


def calibrate_glove(cal_quat):
    theta = np.array(
        [131.65 / 180 * np.pi, 101.61 / 180 * np.pi, 86.19 / 180 * np.pi, 77.31 / 180 * np.pi, 54.52 / 180 * np.pi])
    if hand == 'L':
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
    if hand == 'R':
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
    for i in range(17):
        framem = quat2rotm(cal_quat[REFERENCE_FRAME_ID * 4:REFERENCE_FRAME_ID * 4 + 4])
        rotm = quat2rotm(cal_quat[i * 4:i * 4 + 4])
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


def calculate_feature(quaternions):
    origin = quat2rotm(quaternions[REFERENCE_FRAME_ID * 4:REFERENCE_FRAME_ID * 4 + 4])

    vecSeg = np.array([-1, 0, 0])
    thumb.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)
    index.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)
    middle.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)
    engage.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)
    pinky.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)

    # features extraction
    thumb_fea = np.zeros(NUM_FEATURES_FINGER)
    thumb_fea[0] = get_angle(vecSeg, thumb.orientation[0, :])
    for i in range(1, thumb.numSegment):
        thumb_fea[i] = get_angle(thumb.orientation[i-1, :], thumb.orientation[i, :])

    index_fea = np.zeros(NUM_FEATURES_FINGER)
    index_fea[0] = get_angle(vecSeg, index.orientation[0, :])
    for i in range(1, index.numSegment):
        index_fea[i] = get_angle(index.orientation[i-1, :], index.orientation[i, :])

    middle_fea = np.zeros(NUM_FEATURES_FINGER)
    middle_fea[0] = get_angle(vecSeg, middle.orientation[0, :])
    for i in range(1, middle.numSegment):
        middle_fea[i] = get_angle(middle.orientation[i-1, :], middle.orientation[i, :])

    engage_fea = np.zeros(NUM_FEATURES_FINGER)
    engage_fea[0] = get_angle(vecSeg, engage.orientation[0, :])
    for i in range(1, engage.numSegment):
        engage_fea[i] = get_angle(engage.orientation[i-1, :], engage.orientation[i, :])

    pinky_fea = np.zeros(NUM_FEATURES_FINGER)
    pinky_fea[0] = get_angle(vecSeg, pinky.orientation[0, :])
    for i in range(1, pinky.numSegment):
        pinky_fea[i] = get_angle(pinky.orientation[i-1, :], pinky.orientation[i, :])

    between_fea = np.zeros(NUM_FINGERS - 1)
    between_fea[0] = get_angle(thumb.orientation[0, :], index.orientation[0, :])
    between_fea[1] = get_angle(index.orientation[0, :], middle.orientation[0, :])
    between_fea[2] = get_angle(middle.orientation[0, :], engage.orientation[0, :])
    between_fea[3] = get_angle(engage.orientation[0, :], pinky.orientation[0, :])

    features = np.hstack((thumb_fea, index_fea, middle_fea, engage_fea, pinky_fea, between_fea))
    return features


if __name__ == "__main__":
    t0 = time.clock()

    csvdata = np.genfromtxt('D:\python_program\HandMotionPrimitive\data\withlabel_v2\ID'+str(ID)+'\\'+hand+'_IMU_'+str(trial)+'.csv', delimiter=",")
    info = csvdata[:, 0:2]
    data = csvdata[:, 2:70]

    if data.ndim == 1:  # this is the case when we use 'ground_quats_R or L', where we only have one row
        data = data.reshape(1, data.shape[0])

    calrotm = calibrate_glove(data[0, :])  # Calm[:, :, self.IMUIDs[SegIdx]]
    calpath = 'D:\python_program\HandMotionPrimitive\data\withlabel_v2\ID'+str(ID)+'\\feature\\'+hand+'_IMU_'+str(trial)+'_calrotm.csv'

    calfile = open(calpath, "w", newline='')
    writer = csv.writer(calfile)
    for i in range(17):
        for j in range(3):
            writer.writerow(calrotm[j, :, i])
    calfile.close()

    fea_data = np.zeros((data.shape[0], NUM_FEATURES+1+2))
    feapath = 'D:\python_program\HandMotionPrimitive\data\withlabel_v2\ID'+str(ID)+'\\feature\\'+hand+'_IMU_'+str(trial)+'_fea.csv'
    feafile = open(feapath, "w", newline='')
    writer = csv.writer(feafile)
    for i in range(data.shape[0]):
        features = calculate_feature(data[i, :])
        features = np.hstack((info[i, 0:2], features, [trial]))
        fea_data[i, :] = features
        writer.writerow(features)

    feafile.close()
