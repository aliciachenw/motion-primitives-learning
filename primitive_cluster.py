from sklearn.cluster import Birch
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
import csv
import pickle


# format of the input data:
# [obj, intend, trial, begin time-stamp, end time-stamp, frequency]

n_info = 5
num_sensors = 17


def get_pca(data, n_features):
    info = data[:, 0:n_info]
    fea = data[:, n_info:]

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
        if info[i, 1] == 20:
            marker = 'v'
            color = cmap(2)
        elif info[i, 1] == 21:
            marker = 's'
            color = cmap(3)
        elif info[i, 1] == 23:
            marker = 'd'
            color = cmap(5)
        ax.scatter(fea_pca_3d[i, 0], fea_pca_3d[i, 1], fea_pca_3d[i, 2], c=color, marker=marker)
    plt.title('PCA 3D')
    plt.show()
    print('Plotting is finished!')
    out_data = np.hstack((info, fea_pca))
    return pca, out_data


def clustering_posture(data, n_features, threshold=0.5):
    # BIRCH clustering:
    print('Start Birch clustering!')
    brc = Birch(n_clusters=None, threshold=threshold, compute_labels=True)
    fea_data = data[:, n_info: n_info + n_features]
    t = time()
    brc.fit(fea_data)
    time_ = time() - t
    # labels = brc.labels_
    # n_class = len(np.unique(labels))
    print('Birch clustering is finished!')
    return brc, time_


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
    exp_info = pca_data[:, 0:n_info]
    coord = pca_data[:, n_info:]

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
    seg_method = 'window'

    if hand == 'R':
        handpath = 'right'
    elif hand == 'L':
        handpath = 'left'
    if seg_method == 'window':
        datapath = 'BoW_window_v2.csv'
    elif seg_method == 'zero-crossing':
        datapath = 'BoW_zc_v2.csv'

    datafile = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID'+str(15456)+'\\'+handpath+'\pose\\'+datapath
    # datafile = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(
    #     15456) + '\\' + handpath + '\pose\\' + 'ngram_window.csv'
    data = np.genfromtxt(datafile, delimiter=',')

    if hand == 'L':
        txtpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\left\\primitive\\report.txt'
    elif hand == 'R':
        txtpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\\right\\primitive\\report.txt'
    txtfile = open(txtpath, "w")

    # PCA
    n_features = 8
    pca, pca_data = get_pca(data, n_features)
    print(pca.explained_variance_ratio_)
    s = str(pca.explained_variance_ratio_) + '\n'
    txtfile.write(s)
    total_info = np.sum(pca.explained_variance_ratio_)
    print('total info of pca:', total_info)
    s = 'total info of pca:' + str(total_info) + '\n'
    txtfile.write(s)
    txtfile.write('\n')

    # grid search
    threshold_ = np.linspace(0.05, 1.00, 191)  # 0.1-0.5 for angles
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
        s_score = metrics.silhouette_score(pca_data[:, n_info:], labels, metric='euclidean')
        ch_score = metrics.calinski_harabaz_score(pca_data[:, n_info:], labels)
        s_score_[i] = s_score
        ch_score_[i] = ch_score
        stop_index = i

    ch_score = 0
    for i in range(len(threshold_)):
        if ch_score_[i] > ch_score and n_cluster_[i] >= 10 and n_cluster_[i] <= 50:
            ch_score = ch_score_[i]
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
    s_score = metrics.silhouette_score(pca_data[:, n_info:], labels, metric='euclidean')
    ch_score = metrics.calinski_harabaz_score(pca_data[:, n_info:], labels)
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
        intend = data[j, 1]
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
        pcapath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\left\primitive\pca_model.pickle'
        brcpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\left\primitive\\brc_model.pickle'
    elif hand == 'R':
        pcapath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\\right\primitive\pca_model.pickle'
        brcpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(
            ID) + '\\right\primitive\\brc_model.pickle'
    pcafile = open(pcapath, 'wb')
    brcfile = open(brcpath, 'wb')
    pickle.dump(pca, pcafile)
    pickle.dump(cluster, brcfile)
    pcafile.close()
    brcfile.close()

    # save the center primitive
    if hand == 'L':
        cenpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\left\primitive\center.csv'
    elif hand == 'R':
        cenpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\\right\primitive\center.csv'
    cenfile = open(cenpath, "w", newline='')
    writer = csv.writer(cenfile)

    print("Start plotting the motion primitives!")
    for i in range(n_class):
        for j in range(len(cen_labels)):
            if cen_labels[j] == i:
                cen_ind = j
        cen = center[cen_ind, :]
        nearest_ind = 0
        nearest_distance = 10000
        for j in range(data.shape[0]):
            if labels[j] == i:
                distance = np.linalg.norm(pca_data[j, n_info:]-cen)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_ind = j
                    trial = int(data[nearest_ind, 2])
                    stp = int(data[nearest_ind, 3])
                    endp = int(data[nearest_ind, 4])

        # trialfile = 'D:\python_program\HandMotionPrimitive\data\withlabel_v3\ID' + str(ID) + '\\' + hand + '_IMU_' + str(trial)+'.csv'
        # sensor_data = np.genfromtxt(trialfile, delimiter=",")
        # cen_sensor_data = sensor_data[stp:endp, 2:2+4*num_sensors]
        cen_data = np.hstack(([i], data[nearest_ind, :]))
        writer.writerow(cen_data)
    cenfile.close()

    if hand == 'L':
        labelpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\left\primitive\cluster.csv'
    elif hand == 'R':
        labelpath = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID' + str(ID) + '\\right\primitive\cluster.csv'

    labelfile = open(labelpath, "w", newline='')
    writer = csv.writer(labelfile)
    labels = np.reshape(labels, (len(labels), 1))
    cluster_label = np.hstack((data, labels))
    for t_pose in range(cluster_label.shape[0]):
        writer.writerow(cluster_label[t_pose, :])

    labelfile.close()
    txtfile.close()
