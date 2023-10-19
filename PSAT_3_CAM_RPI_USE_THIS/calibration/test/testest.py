from pycalib.calib import lookat, triangulate, excalib2, pose_registration, rebase
from pycalib.plot import plotCamera
import sys
import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

Nc=2
Np=48
with open('point.pkl', 'rb') as f:
    x=pickle.load(f)

# Camera intrinsics
# K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]
#              ).astype(np.float64)  # VGA camera

K=np.zeros([Nc,3,3])
dist=np.zeros([Nc,5])

with open('client_camera_calib_params.pkl', 'rb') as f:
    calib_params=pickle.load(f)
    rms,cameraMatrix,d,rvect,tvect=calib_params
    K[0]=cameraMatrix
    dist[0]=d
    
with open('server_right_camera_calib_params.pkl', 'rb') as f:
    calib_params=pickle.load(f)
    rms,cameraMatrix,d,rvect,tvect=calib_params
    K[1]=cameraMatrix
    dist[1]=d
    
# with open('server_left_camera_calib_params.pkl', 'rb') as f:
#     calib_params=pickle.load(f)
#     rms,cameraMatrix,d,rvect,tvect=calib_params
#     K[2]=cameraMatrix
#     dist[2]=d


Rt_pairs = dict()
for i in range(Nc-1):
    for j in range(i+1, Nc):
        print(i,j)
        R, t, _, _, _ = excalib2(x[i], x[j], K[i], dist[i], K[j], dist[j])
        Rt_pairs[i, j] = np.hstack((R, t))


with open('stereo_right_camera_calib_params.pkl','rb') as f:
    calib_params=pickle.load(f)
    retval,K[0],dist[0],K[1],dist[1],R,T,E,F,P1,P2 = calib_params
    # Rt_pairs[0, 1][:,3]=T.reshape(-1)
    # Rt_pairs[0, 1][:,0:3] = R
    Rt_pairs[0, 1] = np.hstack((R, T))

# with open('stereo_left_camera_calib_params.pkl','rb') as f:
#     calib_params=pickle.load(f)
#     retval,K[0],dist[0],K[2],dist[2],R,T,E,F,P1,P2 = calib_params
#     Rt_pairs[0, 2][:,3]=T.reshape(-1)
#     # Rt_pairs[0, 2][:,0:3] = R
#     # Rt_pairs[0, 2] = np.hstack((R, T))

# with open('stereo_left_right_camera_calib_params.pkl','rb') as f:
#     calib_params=pickle.load(f)
#     retval,K[1],dist[1],K[2],dist[2],R,T,E,F,P1,P2 = calib_params
#     Rt_pairs[1, 2][:,3]=T.reshape(-1)
#     # Rt_pairs[1, 2][:,0:3] = R
#     # Rt_pairs[1, 2] = np.hstack((R, T))


# Registration
R, t, err_r, err_t = pose_registration(Nc, Rt_pairs)

# Transform to make Camera0 be WCS
R_est = []
t_est = []

for c in reversed(range(Nc)):
    Rx, tx = rebase(R[:3, :3], t[:3], R[3*c:3*c+3, :3], t[3*c:3*c+3])
    R_est.append(Rx)
    t_est.append(tx)
R_est = np.array(R_est[::-1])
t_est = np.array(t_est[::-1])

# This estimation is up-to-scale.  So normalize by the cam1-cam2 distance.
# for c in reversed(range(Nc)):
#     t_est[c] /= np.linalg.norm(t_est[1])

# Projection matrix
P_est = []
for i in range(Nc):
    P_est.append(K[i] @ np.hstack((R_est[i], t_est[i])))
P_est = np.array(P_est)

# Triangulate 3D points
Y_est = []
for i in range(Np):
    y = triangulate(x[0:Nc, i, :].reshape((-1, 2)), P_est)
    Y_est.append(y)
Y_est = np.array(Y_est).T
Y_est = Y_est[:3, :] / Y_est[3, :]

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# ax.set_aspect('equal')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(0, 1)
ax.plot(Y_est[0, :], Y_est[1, :], Y_est[2, :], "o")
cmap = plt.get_cmap("tab10")
for i in range(Nc):
    plotCamera(ax, R_est[i].T, - R_est[i].T @ t_est[i], cmap(i), 0.05)
    
corners=np.array([[Y_est[0,0],Y_est[1,0],Y_est[2,0]],[Y_est[0,7],Y_est[1,7],Y_est[2,7]],[Y_est[0,47],Y_est[1,47],Y_est[2,47]],[Y_est[0,40],Y_est[1,40],Y_est[2,40]]])

ax.plot(corners[:,0],corners[:,1],corners[:,2],'-')

fig.show()