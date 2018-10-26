# Converting data into RGB heatmap

import numpy as np


data = np.linspace(0,10,100)
data = np.asarray(np.meshgrid(data,data))
data = data[0]


#####

dataShape = data.shape

data = data.reshape((-1,1))

# normalizing the data

alpha = np.min(np.min(data))
beta = np.max(np.max(data))
gamma = beta - alpha
data = data - alpha
data = data / gamma


# Defining Transissions

Rpx = np.array([0,      .125,       .38,        .62,       .88,         1])
Gpx = Rpx; Bpx = Rpx

Rpy = np.array([0,        0,       .00,          1,          1,         .5])
Gpy = np.array([0,        0,         1,          1,          0,          0])
Bpy = np.array([.5,       1,         1,          0,          0,          0])



RGBmap3D = np.zeros((1,data.size,3))


for i in range(data.size):
    if data[i] <= Rpx[1]:
        RGBmap3D[0,i,:] = [(np.diff(Rpy[0:2]) / np.diff(Rpx[0:2])) * (data[i] - Rpx[0]) + Rpy[0],
            (np.diff(Gpy[0:2]) / np.diff(Gpx[0:2])) * (data[i] - Gpx[0]) + Gpy[0],
            (np.diff(Gpy[0:2]) / np.diff(Gpx[0:2])) * (data[i] - Gpx[0]) + Gpy[0]]

    elif data[i] <= Rpx[2]:
        RGBmap3D[0,i,:] = [(np.diff(Rpy[1:3]) / np.diff(Rpx[1:3])) * (data[i] - Rpx[1]) + Rpy[1],
            (np.diff(Gpy[1:3]) / np.diff(Gpx[1:3])) * (data[i] - Gpx[1]) + Gpy[1],
            (np.diff(Gpy[1:3]) / np.diff(Gpx[1:3])) * (data[i] - Gpx[1]) + Gpy[1]]

    elif data[i] <= Rpx[3]:
        RGBmap3D[0,i,:] = [(np.diff(Rpy[2:4]) / np.diff(Rpx[2:4])) * (data[i] - Rpx[2]) + Rpy[2],
            (np.diff(Gpy[2:4]) / np.diff(Gpx[2:4])) * (data[i] - Gpx[2]) + Gpy[2],
            (np.diff(Gpy[2:4]) / np.diff(Gpx[2:4])) * (data[i] - Gpx[2]) + Gpy[2]]

    elif data[i] <= Rpx[4]:
        RGBmap3D[0,i,:] = [(np.diff(Rpy[3:5]) / np.diff(Rpx[3:5])) * (data[i] - Rpx[3]) + Rpy[3],
            (np.diff(Gpy[3:5]) / np.diff(Gpx[3:5])) * (data[i] - Gpx[3]) + Gpy[3],
            (np.diff(Gpy[3:5]) / np.diff(Gpx[3:5])) * (data[i] - Gpx[3]) + Gpy[3]]

    elif data[i] <= Rpx[5]:
        RGBmap3D[0,i,:] = [(np.diff(Rpy[4:6]) / np.diff(Rpx[4:6])) * (data[i] - Rpx[4]) + Rpy[4],
            (np.diff(Gpy[4:6]) / np.diff(Gpx[4:6])) * (data[i] - Gpx[4]) + Gpy[4],
            (np.diff(Gpy[4:6]) / np.diff(Gpx[4:6])) * (data[i] - Gpx[4]) + Gpy[4]]

RGBmap = np.reshape(RGBmap3D,(dataShape[0],dataShape[1],3))


print('finished')