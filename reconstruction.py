import os
import pickle
import numpy as np
import sys
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as an
import imageio
import matplotlib.image as mpimg

ID = 15456
hand = 'L'
trial = 1

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("new folder created!\n")
    else:
        print("folder already existed!\n")


if hand == 'L':
    handpath = 'left'
if hand == 'R':
    handpath = 'right'

clusterfile = 'D:\python_program\HandMotionPrimitive\Clustering\\results_v4\ID'+str(ID)+'\\'+handpath+'\pose\\cluster.csv'
data = np.genfromtxt(clusterfile, delimiter=",")
flag = 0
for i in range(data.shape[0]):
    if data[i, 3] == trial:
        if flag == 0:
            trialdata = data[i, :]
            flag = 1
        else:
            trialdata = np.vstack((trialdata, data[i, :]))  # extract trial

stp = int(trialdata[0, 4])
endp = int(trialdata[trialdata.shape[0]-1, 4])

# reconstruction
foldpath = 'D:\python_program\HandMotionPrimitive\FinalResults\\ID'+str(ID)+'_'+hand+'_trial_'+str(trial)+'_genfromclus'
mkdir(foldpath)
centerpath = 'D:\python_program\HandMotionPrimitive\FinalResults\\visualization\\'+handpath+'\center\\'

for i in range(trialdata.shape[0]):
    cluster = int(trialdata[i, 2])
    oldpath = centerpath+'Figure_'+str(cluster)+'.png'
    img = str(i)
    img = img.zfill(5)
    newpath = foldpath+'\Image_'+img+'.png'
    shutil.copy(oldpath, newpath)

imagelist = os.listdir(foldpath)
clusfile = 'D:\python_program\HandMotionPrimitive\FinalResults\\visualization\ID'+str(ID)+'_'+hand+'_trial_'+str(trial)+'_genfromclus.gif'
frames = []
for image_name in imagelist:
    frames.append(mpimg.imread(foldpath+'\\'+image_name))


def update(frame_number):
    return plt.imshow(frames[frame_number])


fig = plt.figure()
fig.clf()
ani = an.FuncAnimation(fig, update, interval=50, blit=False)
plt.axis('off')
plt.show()
