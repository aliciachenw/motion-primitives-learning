import numpy as np
import os
import time
import csv

# update 8/7/2018
# Calculated the joint angles anatomically


hand = 'L'
ID = 15456
trial = 5


class SGloveFingers:
    def __init__(self, _numSeg, _IMUIDs, startP):
        self.IMUIDs = _IMUIDs[startP:startP + _numSeg]
        self.IMURots = np.zeros([_numSeg, 3, 3])
        self.orientation = np.zeros([_numSeg, 3])
        self.numSegment = _numSeg

    def update_info(self, SegIdx, quaternions, Pos, Calm, origin):
        qIdx = 4 * self.IMUIDs[SegIdx]
        rotm = quat2rotm(quaternions[qIdx:qIdx + 4])
        rotm = np.matmul(rotm, Calm[:, :, self.IMUIDs[SegIdx]])
        rotm = np.matmul(rotm, np.linalg.inv(origin))
        self.IMURots[SegIdx, :, :] = rotm
        self.orientation[SegIdx, :] = np.matmul(rotm, Pos)

    def update_infoAllSeg(self, quaternions, segPos, Calm, origin):
        for i in range(self.numSegment):
            self.update_info(i, quaternions, segPos, Calm, origin)


DATA_LEN = 4
REFERENCE_FRAME_ID = 15
SECOND_PALM_ID = 16
NUM_FINGERS = 5
NUM_FEATURES_FINGER = 4
NUM_FEATURES = NUM_FEATURES_FINGER*NUM_FINGERS


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
    return rot3M


if hand == 'R':
    IMUIDs = np.array(
        [6, 7, 8, 3, 4, 5, 12, 13, 14, 9, 10, 11, 0, 1, 2])  # thumb root to tip -> index root to tip -> ... {1,2,3,...}
elif hand == 'L':
    IMUIDs = np.array(
        [6, 7, 8, 0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5])  # thumb root to tip -> index root to tip -> ... {1,2,3,...}


thumb = SGloveFingers(3, IMUIDs, 0)
index = SGloveFingers(3, IMUIDs, 3)
middle = SGloveFingers(3, IMUIDs, 6)
engage = SGloveFingers(3, IMUIDs, 9)
pinky = SGloveFingers(3, IMUIDs, 12)

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


def calibrate_glove(cal_quat):
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


def get_localXYZ(mat):
    if hand == 'R':
        local_mat = np.zeros([3, 3])
        local_mat[0, :] = mat[2, :]
        for i in range(3):
            local_mat[0, i] = -local_mat[0, i]
        local_mat[1, :] = mat[0, :]
        local_mat[2, :] = mat[1, :]
        for i in range(3):
            local_mat[2, i] = -local_mat[2, i]

    if hand == 'L':
        local_mat = np.zeros([3, 3])
        local_mat[0, :] = mat[2, :]
        local_mat[1, :] = mat[0, :]
        for i in range(3):
            local_mat[1, i] = -local_mat[1, i]
        local_mat[2, :] = mat[1, :]
        for i in range(3):
            local_mat[2, i] = -local_mat[2, i]

    return local_mat


def get_Euler_angle(rotm):
    # ZXY
    theta = np.zeros(3)
    for i in range(3):
        n = rotm[i, 0] ** 2 + rotm[i, 1] ** 2 + rotm[i, 2] ** 2
        n = np.sqrt(n)
        rotm[i, 0] = rotm[i, 0] / n
        rotm[i, 1] = rotm[i, 1] / n
        rotm[i, 2] = rotm[i, 2] / n
    if rotm[2, 1] >= 1.000:
        theta[0] = -np.pi / 2
    elif rotm[2, 1] <= -1.0000:
        theta[0] = np.pi / 2
    else:
        theta[0] = np.arcsin(-rotm[2, 1])  # theta_x

    if rotm[2, 0] / np.cos(theta[0]) >= 1.000:
        theta[1] = np.pi / 2
    elif rotm[2, 0] / np.cos(theta[0]) <= -1.000:
        theta[1] = -np.pi / 2
    else:
        theta[1] = np.arcsin(rotm[2, 0] / np.cos(theta[0]))  # theta_y

    if rotm[0, 1] / np.cos(theta[0]) >= 1.000:
        theta[2] = np.pi / 2
    elif rotm[0, 1] / np.cos(theta[0]) <= -1.000:
        theta[2] = -np.pi / 2
    else:
        theta[2] = np.arcsin(rotm[0, 1] / np.cos(theta[0]))  # theta_z

    theta[0] = -theta[0]
    theta[2] = -theta[2]
    return theta


def calculate_feature(quaternions):
    origin = quat2rotm(quaternions[REFERENCE_FRAME_ID * 4:REFERENCE_FRAME_ID * 4 + 4])

    vecSeg = np.array([-1, 0, 0])
    thumb.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)
    index.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)
    middle.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)
    engage.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)
    pinky.update_infoAllSeg(quaternions, vecSeg, calrotm, origin)

    # features extraction
    # TODO: Anatomical Angles (Euler Angles?)
    # reference: ISB recommendation
    # define palm coordinate system (also mcp coordinate system)
    if hand == 'R':
        palm_local = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    elif hand == 'L':
        palm_local = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    # thumb
    # abduction/adduction for 1st MCP
    thumb_fea = np.zeros(NUM_FEATURES_FINGER)
    thumb_mcp = get_localXYZ(thumb.IMURots[0, :, :])
    relativemat = np.matmul(thumb_mcp, np.linalg.inv(palm_local))
    euler = get_Euler_angle(relativemat)
    thumb_fea[0] = euler[0] / np.pi
    # flexion/extension
    thumb_fea[1] = euler[2] / np.pi
    for i in range(1, thumb.numSegment):
        framemat = get_localXYZ(thumb.IMURots[i-1, :, :])
        thumb_joint = get_localXYZ(thumb.IMURots[i, :, :])
        relativemat = np.matmul(thumb_joint, np.linalg.inv(framemat))
        euler = get_Euler_angle(relativemat)
        thumb_fea[i+1] = euler[2] / np.pi

    # index
    # abduction/adduction for 1st MCP
    index_fea = np.zeros(NUM_FEATURES_FINGER)
    index_mcp = get_localXYZ(index.IMURots[0, :, :])
    relativemat = np.matmul(index_mcp, np.linalg.inv(palm_local))
    euler = get_Euler_angle(relativemat)
    index_fea[0] = euler[0] / np.pi
    # flexion/extension
    index_fea[1] = euler[2] / np.pi
    for i in range(1, thumb.numSegment):
        framemat = get_localXYZ(index.IMURots[i - 1, :, :])
        index_joint = get_localXYZ(index.IMURots[i, :, :])
        relativemat = np.matmul(index_joint, np.linalg.inv(framemat))
        euler = get_Euler_angle(relativemat)
        index_fea[i + 1] = euler[2] / np.pi

    # middle
    middle_fea = np.zeros(NUM_FEATURES_FINGER)
    middle_mcp = get_localXYZ(middle.IMURots[0, :, :])
    relativemat = np.matmul(middle_mcp, np.linalg.inv(palm_local))
    euler = get_Euler_angle(relativemat)
    middle_fea[0] = euler[0] / np.pi
    # flexion/extension
    middle_fea[1] = euler[2] / np.pi
    for i in range(1, thumb.numSegment):
        framemat = get_localXYZ(middle.IMURots[i - 1, :, :])
        middle_joint = get_localXYZ(middle.IMURots[i, :, :])
        relativemat = np.matmul(middle_joint, np.linalg.inv(framemat))
        euler = get_Euler_angle(relativemat)
        middle_fea[i + 1] = euler[2] / np.pi

    # engage
    engage_fea = np.zeros(NUM_FEATURES_FINGER)
    engage_mcp = get_localXYZ(engage.IMURots[0, :, :])
    relativemat = np.matmul(engage_mcp, np.linalg.inv(palm_local))
    euler = get_Euler_angle(relativemat)
    engage_fea[0] = euler[0] / np.pi
    # flexion/extension
    engage_fea[1] = euler[2] / np.pi
    for i in range(1, thumb.numSegment):
        framemat = get_localXYZ(engage.IMURots[i - 1, :, :])
        engage_joint = get_localXYZ(engage.IMURots[i, :, :])
        relativemat = np.matmul(engage_joint, np.linalg.inv(framemat))
        euler = get_Euler_angle(relativemat)
        engage_fea[i + 1] = euler[2] / np.pi

    # pinky
    pinky_fea = np.zeros(NUM_FEATURES_FINGER)
    pinky_mcp = get_localXYZ(pinky.IMURots[0, :, :])
    relativemat = np.matmul(pinky_mcp, np.linalg.inv(palm_local))
    euler = get_Euler_angle(relativemat)
    pinky_fea[0] = euler[0] / np.pi
    # flexion/extension
    pinky_fea[1] = euler[2] / np.pi
    for i in range(1, thumb.numSegment):
        framemat = get_localXYZ(pinky.IMURots[i - 1, :, :])
        pinky_joint = get_localXYZ(pinky.IMURots[i, :, :])
        relativemat = np.matmul(pinky_joint, np.linalg.inv(framemat))
        euler = get_Euler_angle(relativemat)
        pinky_fea[i + 1] = euler[2] / np.pi

    # return features
    features = np.hstack((thumb_fea, index_fea, middle_fea, engage_fea, pinky_fea))
    return features


if __name__ == "__main__":
    t0 = time.clock()

    csvdata = np.genfromtxt('D:\python_program\HandMotionPrimitive\data\withlabel_v3\ID'+str(ID)+'\\'+hand+'_IMU_'+str(trial)+'.csv', delimiter=",")
    info = csvdata[:, 0:2]
    data = csvdata[:, 2:70]

    if data.ndim == 1:  # this is the case when we use 'ground_quats_R or L', where we only have one row
        data = data.reshape(1, data.shape[0])

    calrotm = calibrate_glove(data[0, :])  # Calm[:, :, self.IMUIDs[SegIdx]]
    calpath = 'D:\python_program\HandMotionPrimitive\data\withlabel_v3\ID'+str(ID)+'\\feature\\'+hand+'_IMU_'+str(trial)+'_calrotm.csv'

    calfile = open(calpath, "w", newline='')
    writer = csv.writer(calfile)
    for i in range(17):
        for j in range(3):
            writer.writerow(calrotm[j, :, i])
    calfile.close()

    fea_data = np.zeros((data.shape[0], NUM_FEATURES+1+2))
    feapath = 'D:\python_program\HandMotionPrimitive\data\withlabel_v3\ID'+str(ID)+'\\feature\\'+hand+'_IMU_'+str(trial)+'_fea.csv'
    feafile = open(feapath, "w", newline='')
    writer = csv.writer(feafile)
    for i in range(data.shape[0]):
        features = calculate_feature(data[i, :])
        features = np.hstack((info[i, 0:2], features, [trial]))
        fea_data[i, :] = features
        writer.writerow(features)

    feafile.close()
