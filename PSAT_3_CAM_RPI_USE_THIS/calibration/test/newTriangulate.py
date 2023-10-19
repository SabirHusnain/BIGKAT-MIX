from myCalib import lookat, triangulate, excalib2, pose_registration, rebase, excalibN
from pycalib.plot import plotCamera
import sys
import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
import pickle

Nc=3
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
    
with open('server_left_camera_calib_params.pkl', 'rb') as f:
    calib_params=pickle.load(f)
    rms,cameraMatrix,d,rvect,tvect=calib_params
    K[2]=cameraMatrix
    dist[2]=d

camIdx=np.array(range(0,Nc))
pntIdx=np.array(range(0,Np))

out = excalibN(K, dist, camIdx, pntIdx, x)
R_est, t_est, Y_est, PIDs_ok=out

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