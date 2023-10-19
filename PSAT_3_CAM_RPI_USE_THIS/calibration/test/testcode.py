import sys, os, cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pycalib.plot import plotCamera
import pycalib.calib as pcalib

import pickle

with open('point.pkl', 'rb') as f:
    x=pickle.load(f)
    
Nc=3
Np=48
K=np.zeros([3,3,3])
dist=np.zeros([3,5])

with open('client_camera_calib_params.pkl', 'rb') as f:
    calib_params=pickle.load(f)
    rms,cameraMatrix,d,rvect,tvect=calib_params
    K[0]=cameraMatrix
    dist[0]=d.reshape([-1])
    
with open('server_right_camera_calib_params.pkl', 'rb') as f:
    calib_params=pickle.load(f)
    rms,cameraMatrix,d,rvect,tvect=calib_params
    K[1]=cameraMatrix
    dist[1]=d.reshape([-1])
    
with open('server_left_camera_calib_params.pkl', 'rb') as f:
    calib_params=pickle.load(f)
    rms,cameraMatrix,d,rvect,tvect=calib_params
    K[2]=cameraMatrix
    dist[2]=d.reshape([-1])
    
RtEsX_pairs = dict()
for i in range(Nc-1):
    for j in range(i+1, Nc):
        R, t, E, status, X = pcalib.excalib2(x[i], x[j], K[i], dist[i], K[j], dist[j])
        RtEsX_pairs[i, j] = np.hstack((R, t))

# Registration
R, t, err_r, err_t = pcalib.pose_registration(Nc, RtEsX_pairs)

# Transform to make Camera0 be WCS
R_est = []
t_est = []

for c in reversed(range(Nc)):
    Rx, tx = pcalib.rebase(R[:3, :3], t[:3], R[3*c:3*c+3, :3], t[3*c:3*c+3])
    R_est.append(Rx)
    t_est.append(tx)
R_est = np.array(R_est[::-1])
t_est = np.array(t_est[::-1])

# This estimation is up-to-scale.  So normalize by the cam1-cam2 distance.
for c in reversed(range(Nc)):
    t_est[c] /= np.linalg.norm(t_est[1])
    
# Projection matrix
P_est = []
for i in range(Nc):
    P_est.append(K[i] @ np.hstack((R_est[i], t_est[i])))
P_est = np.array(P_est)

# Triangulate 3D points
Y_est = []
for i in range(x.shape[1]):
    try:
        y = pcalib.triangulate(x[:,i,:].reshape((-1,2)), P_est)
    except:
        y=np.array([np.nan,np.nan,np.nan,1])
    Y_est.append(y)
Y_est = np.array(Y_est).T
Y_est = Y_est[:3,:] / Y_est[3,:]

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# ax.set_aspect('equal')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(0, 1)
# ax.plot(Y_est[0,:], Y_est[1,:], Y_est[2,:], "o")
cmap = plt.get_cmap("tab10")
for i in range(Nc):
    plotCamera(ax, R_est[i].T, - R_est[i].T @ t_est[i], cmap(i), 0.05)
fig.show()