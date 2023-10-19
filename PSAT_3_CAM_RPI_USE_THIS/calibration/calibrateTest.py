import sys, os, cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pycalib.plot import plotCamera
# Chessboard configuration
rows = 6   # Number of corners (not cells) in row
cols = 8  # Number of corners (not cells) in column
size = 35 # Physical size of a cell (the distance between neighrboring corners). Any positive number works.

# Input images capturing the chessboard above
input_files = 'client/*.tiff'

# plotCamera() config
plot_mode   = 0    # 0: fixed camera / moving chessboard,  1: fixed chessboard, moving camera
plot_range  = 2000 # target volume [-plot_range:plot_range]
camera_size = 100  # size of the camera in plot
X_W = np.empty([rows * cols, 3], dtype=np.float32)
for i_row in range(0, rows):
    for i_col in range(0, cols):
        X_W[i_row*cols+i_col] = np.array([size*i_col, size*i_row, 0], dtype=np.float32)
print(X_W)
Xs_W = []
xs_I = []

for image_path in glob(input_files): # for each chessboard image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)      # load the image
    found, x_I = cv2.findChessboardCorners(image, (cols, rows)) # detech the chess corners

    if found: # if found
        term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        x_I_sub = cv2.cornerSubPix(image, x_I, (5,5), (-1,-1), term) # refine the corner positions
        Xs_W.append(X_W)     # the chess corner in 3D
        xs_I.append(x_I_sub) # is projected to this 2D position
    else:     # if not found
        print("Detection failed :", image_path)
        continue
# rep, K, d, rvec, tvec = cv2.calibrateCamera(Xs_W, xs_I, (image.shape[1], image.shape[0]), None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)

import pickle
with open('client_camera_calib_params.pkl','rb') as f:
    data=pickle.load(f)
    rep, K, d, rvec, tvec=data

print("Intrinsic parameter K = ", K)
print("Distortion parameters d = (k1, k2, p1, p2, k3) = ", d)

fig_in = plt.figure()
ax_in = Axes3D(fig_in, auto_add_to_figure=False)
fig_in.add_axes(ax_in)

ax_in.set_xlim(-plot_range, plot_range)
ax_in.set_ylim(-plot_range, plot_range)
ax_in.set_zlim(-plot_range, plot_range)

if plot_mode == 0: # fixed camera = plot in CCS
    
    plotCamera(ax_in, np.eye(3), np.zeros((1,3)), "b", camera_size) # camera is at (0,0,0)
    
    for i_ex in range(len(rvec)):
        X_C = np.zeros((X_W.shape))
        for i_x in range(X_W.shape[0]):
            R_w2c = cv2.Rodrigues(rvec[i_ex])[0] # convert to the rotation matrix
            t_w2c = tvec[i_ex].reshape(3)
            X_C[i_x,:] = R_w2c.dot(X_W[i_x,:]) + t_w2c # Transform chess corners in WCS to CCS
                
        ax_in.plot(X_C[:,0], X_C[:,1], X_C[:,2], ".") # plot chess corners in CCS
        
elif plot_mode == 1: # fixed chessboard = plot in WCS
    
    for i_ex in range(len(rvec)):
        R_c2w = np.linalg.inv(cv2.Rodrigues(rvec[i_ex])[0]) #　世界座標系におけるカメラの向き
        t_c2w = -R_c2w.dot(tvec[i_ex]).reshape((1,3)) # 世界座標系におけるカメラの位置
        
        plotCamera(ax_in, R_c2w, t_c2w, "b", camera_size)
        
    ax_in.plot(X_W[:,0], X_W[:,1], X_W[:,2], ".")

fig_in.show()