from sklearn.cluster import Birch
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
from glove_viz_v2 import plot_posture
import csv
import pickle
 
# update 7/26/2018:
# intent 22 (finding) is discarded; end of each intend is ignored;
# add grid search
# update 8/3/2018:
# new features & visualization (withlabel_v2, glove_viz_v2)
# update 8/6/2018:
# save model


num_sensors = 17
num_features = 19


# load trials
def load_trials(ID, hand, feature=True):

    if feature:
        path = 'D:\python_program\HandMotionPrimitive\data\withlabel_v2\ID' + str(ID) + '\\feature'
    else:
        path = 'D:\python_program\HandMotionPrimitive\data\withlabel_v2\ID' + str(ID)
    file = os.listdir(path)

    flag = 0
    if hand == 'L' or hand == 'R':
        for f in file:
            fname = os.path.splitext(f)[0]
            if (hand in fname) and ('fea' in fname):
                filename = path + '\\' + fname + '.csv'
                trial_data = np.genfromtxt(filename, delimiter=",")
                for i in range(trial_data.shape[0]):
                    if trial_data[i, 1] != 22 and trial_data[i, 1] != 40:  # discard the start & finding phase
                        if i < trial_data.shape[0]-20:  # not at the end
                            if trial_data[i, 1] == trial_data[i+20, 1]:  # discard the transition phase
                                if flag == 0:
                                    out_data = np.hstack((trial_data[i, :], [i]))
                                    out_data = np.reshape(out_data, (1, len(out_data)))
                                    flag = 1
                                else:
                                    temp = np.hstack((trial_data[i, :], [i]))
                                    temp = np.reshape(temp, (1, len(temp)))
                                    out_data = np.concatenate((out_data, temp))
                        else:
                            if flag == 0:
                                out_data = np.hstack((trial_data[i, :], [i]))
                                out_data = np.reshape(out_data, (1, len(out_data)))
                                flag = 1
                            else:
                                temp = np.hstack((trial_data[i, :], [i]))
                                temp = np.reshape(temp, (1, len(temp)))
                                out_data = np.concatenate((out_data, temp))
    else:
        print('wrong hand!')
        out_data = []
    return out_data


def get_pca(data, n_features, feature=True):
    exp_info = data[:, 0:2]
    trial_info = data[:, data.shape[1]-2:data.shape[1]-1]
    # info[:, 0]: obj number  (0: cylinder, 1: box, 2: disk, 3: ball)
    # info[:, 1]: intend number (20: explore, 21: excavate, 22: find, 23: retrieve)
    # info[:, data.shape[1]-2] : trial number
    # info[:, data.shape[1]-1] : time stamp in trial
    if feature:
        fea = data[:, 2:2+num_features]
    else:
        fea = data[:, 2:2+4*num_sensors]
    # Prominent component analysis:
    print('Start PCA!')
    pca = PCA(n_components=n_features)
    pca.fit(fea)
    fea_pca = pca.transform(fea)
    print('PCA is finished!')
    print('Start plotting result of PCA!')
    
    # 3D-PCA visualization (Higher dimensions??)
    fea_pca_3d = fea_pca[:, 0:3]
    fig = plt.figure()
    cmap = plt.cm.get_cmap('Paired')
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(fea_pca_3d.shape[0]):
        if exp_info[i, 1] == 20:
            marker = 'v'
            color = cmap(2)
        elif exp_info[i, 1] == 21:
            marker = 's'
            color = cmap(3)
        elif exp_info[i, 1] == 23:
            marker = 'd'
            color = cmap(5)
        ax.scatter(fea_pca_3d[i, 0], fea_pca_3d[i, 1], fea_pca_3d[i, 2], c=color, marker=marker)
    plt.title('PCA 3D')
    plt.show()
    print('Plotting is finished!')
    out_data = np.hstack((exp_info, fea_pca, trial_info))
    return pca, out_data


def clustering_posture(data, n_features, threshold=0.5):
    # BIRCH clustering:
    print('Start Birch clustering!')
    brc = Birch(n_clusters=None, threshold=threshold, compute_labels=True)
    fea_data = data[:, 2:2+n_features]
    t = time()
    brc.fit(fea_data)
    time_ = time() - t
    # labels = brc.labels_
    # n_class = len(np.unique(labels))
    print('Birch clustering is finished!')
    return brc, time_


    # clustering results visulization
def print_cluster(pca_data, cluster):
    print('Start plotting result of Birch clustering!')

    labels = cluster.labels_
    cen_labels = cluster.subcluster_labels_
    n_class = len(np.unique(cen_labels))

    fig = plt.figure()
    cmap = plt.cm.get_cmap()
    color_index = np.linspace(0, 1, n_class)
    fig.clf()
    ax2 = fig.add_subplot(111, projection='3d')
    exp_info = pca_data[:, 0:2]
    coord = pca_data[:, 2:5]

    for i in range(coord.shape[0]):
        if exp_info[i, 1] == 20:
            marker = 'v'
        elif exp_info[i, 1] == 21:
            marker = 's'
        elif exp_info[i, 1] == 23:
            marker = 'd'
        color = cmap(color_index[labels[i]])
        ax2.scatter(coord[i, 0], coord[i, 1], coord[i, 2], c=color, marker=marker)
    plt.title('Birch 3D')
    plt.show()
    print('Plotting is finished!')


if __name__ == "__main__":
    ID = 15456
    hand = 'L'
    feature = False
    pose_data = load_trials(ID, hand, feature=feature)

    if hand == 'L':
        txtpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\left\\report.txt'
    elif hand == 'R':
        txtpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\\right\\report.txt'
    txtfile = open(txtpath, "w")

    # PCA
    n_features = 8
    pca, pca_data = get_pca(pose_data, n_features, feature=True)
    print(pca.explained_variance_ratio_)
    s = str(pca.explained_variance_ratio_) + '\n'
    txtfile.write(s)
    total_info = np.sum(pca.explained_variance_ratio_)
    print('total info of pca:', total_info)
    s = 'total info of pca:' + str(total_info) + '\n'
    txtfile.write(s)
    txtfile.write('\n')

    # grid search
    threshold_ = np.linspace(0.10, 0.50, 81)  # 0.1-0.5 for angles
    s_score_ = np.zeros(threshold_.shape)
    ch_score_ = np.zeros(threshold_.shape)
    n_cluster_ = np.zeros(threshold_.shape)
    for i in range(len(threshold_)):
        cluster, t = clustering_posture(pca_data, n_features, threshold_[i])
        labels = cluster.labels_
        n_cluster_[i] = len(np.unique(labels))
        if n_cluster_[i] == 1:
            stop_index = i
            for j in range(i, len(threshold_)):
                n_cluster_[j] = 1
            break
        s_score = metrics.silhouette_score(pca_data[:, 2:2+n_features], labels, metric='euclidean')
        ch_score = metrics.calinski_harabaz_score(pca_data[:, 2:2+n_features], labels)
        s_score_[i] = s_score
        ch_score_[i] = ch_score
        stop_index = i

    ch_score = 0
    for i in range(len(threshold_)):
        if ch_score_[i] > ch_score and n_cluster_[i] >= 20:
            index = i

    threshold = threshold_[index]
    print('best threshold:', threshold)
    s = 'best threshold:' + str(threshold) + '\n'
    txtfile.write(s)
    fig = plt.figure()
    plt.plot(threshold_[0:stop_index], s_score_[0:stop_index])
    plt.title('s_score')
    plt.show()
    fig = plt.figure()
    plt.plot(threshold_[0:stop_index], ch_score_[0:stop_index])
    plt.title('ch_score')
    plt.show()
    fig = plt.figure()
    plt.plot(threshold_[0:stop_index], n_cluster_[0:stop_index])
    plt.title('num of cluster')
    plt.show()

    # evaluate pca & clustering
    cluster, t = clustering_posture(pca_data, n_features, threshold)
    labels = cluster.labels_
    s_score = metrics.silhouette_score(pca_data[:, 2:2 + n_features], labels, metric='euclidean')
    ch_score = metrics.calinski_harabaz_score(pca_data[:, 2:2 + n_features], labels)
    center = cluster.subcluster_centers_
    cen_labels = cluster.subcluster_labels_
    n_class = len(np.unique(cen_labels))
    print_cluster(pca_data, cluster)

    print('number of clusters:', n_class)
    s = 'number of clusters:' + str(n_class) + '\n'
    txtfile.write(s)
    print('silhouette score:', s_score)
    s = 'silhouette score:' + str(s_score) + '\n'
    txtfile.write(s)
    print('calinski harabaz score:', ch_score)
    s = 'calinski harabaz score:' + str(ch_score) + '\n'
    txtfile.write(s)
    txtfile.write('\n')

    # statistics of the frequency of 4 intends in clusters
    fre_mat = np.zeros((n_class, 3))
    for j in range(len(labels)):
        intend = pose_data[j, 1]
        if intend == 20:
            b = 0
        elif intend == 21:
            b = 1
        elif intend == 23:
            b = 2
        a = labels[j]
        fre_mat[a, b] = fre_mat[a, b] + 1
    for i in range(fre_mat.shape[0]):
        s = str(fre_mat[i, :]) + '\n'
        txtfile.write(s)
    txtfile.write('\n')
    fig = plt.figure()
    name_list = ['explore', 'excavate', 'retrieve']
    fre_sum = np.sum(fre_mat, axis=0)
    for i in range(fre_mat.shape[0]):
        for j in range(fre_mat.shape[1]):
            fre_mat[i, j] = fre_mat[i, j]/fre_sum[j]
    total_width = 1
    width = (total_width-0.1) / n_class
    cmap = plt.cm.get_cmap()
    color_index = np.linspace(0, 1, n_class)
    color = cmap(color_index[0])
    x = list(range(len(name_list)))
    plt.bar(x, fre_mat[0, :], width=width, label='0', fc=color, tick_label=name_list)
    for i in range(1, n_class):
        for j in range(3):
            x[j] = x[j] + width
        color = cmap(color_index[i])
        plt.bar(x, fre_mat[i, :], width=width, label=str(i), fc=color)
    plt.legend()
    plt.show()

    if hand == 'L':
        pcapath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\left\pca_model.pickle'
        brcpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\left\\brc_model.pickle'
    elif hand == 'R':
        pcapath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\\right\pca_model.pickle'
        brcpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(
            ID) + '\\right\\brc_model.pickle'
    pcafile = open(pcapath, 'wb')
    brcfile = open(brcpath, 'wb')
    pickle.dump(pca, pcafile)
    pickle.dump(cluster, brcfile)
    pcafile.close()
    brcfile.close()


    # visualize the center pose
    if hand == 'L':
        cenpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\left\center.csv'
    elif hand == 'R':
        cenpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\\right\center.csv'
    cenfile = open(cenpath, "w", newline='')
    writer = csv.writer(cenfile)

    print("Start plotting the static postures!")
    for i in range(n_class):
        for j in range(len(cen_labels)):
            if cen_labels[j] == i:
                cen_ind = j
        cen = center[cen_ind, :]
        nearest_ind = 0
        nearest_distance = 10000
        for j in range(pose_data.shape[0]):
            if labels[j] == i:
                distance = np.linalg.norm(pca_data[j, 2:2+n_features]-cen)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_ind = j
                    trial = int(pose_data[nearest_ind, pose_data.shape[1]-2])
                    time_stamp = int(pose_data[nearest_ind, pose_data.shape[1]-1])

        trialfile = 'D:\python_program\HandMotionPrimitive\data\withlabel_v2\ID' + str(ID) + '\\' + hand + '_IMU_' + str(trial)+'.csv'
        sensor_data = np.genfromtxt(trialfile, delimiter=",")
        cen_sensor_data = sensor_data[time_stamp, 2:2+4*num_sensors]
        plot_posture(cen_sensor_data, ID, trial, hand, i)
        cen_data = np.hstack(([i], cen_sensor_data, [trial], [time_stamp]))
        writer.writerow(cen_data)
    cenfile.close()

    if hand == 'L':
        labelpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\left\cluster.csv'
    elif hand == 'R':
        labelpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v3\ID' + str(ID) + '\\right\cluster.csv'

    labelfile = open(labelpath, "w", newline='')
    writer = csv.writer(labelfile)
    labels = np.reshape(labels, (len(labels), 1))
    cluster_label = np.hstack((pose_data[:, 0:2], labels, pose_data[:, pose_data.shape[1]-2:pose_data.shape[1]]))
    for t_pose in range(cluster_label.shape[0]):
        writer.writerow(cluster_label[t_pose, :])

    labelfile.close()
    txtfile.close()





