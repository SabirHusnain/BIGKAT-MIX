from pycalib.calib import lookat, triangulate, excalib2, pose_registration, rebase
from pycalib.plot import plotCamera
import sys
import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
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
for c in reversed(range(Nc)):
    t_est[c] /= np.linalg.norm(t_est[1])

# Projection matrix
P_est = []
for i in range(Nc):
    P_est.append(K[i] @ np.hstack((R_est[i], t_est[i])))
P_est = np.array(P_est)

# Triangulate 3D points
Y_est_1 = []
for i in range(Np):
    y = triangulate(x[0:Nc, i, :].reshape((-1, 2)), P_est)
    Y_est_1.append(y)
Y_est_1 = np.array(Y_est_1).T
Y_est_1 = Y_est_1[:3, :] / Y_est_1[3, :]

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# ax.set_aspect('equal')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(0, 1)
ax.plot(Y_est_1[0, :], Y_est_1[1, :], Y_est_1[2, :], "o", label='C-R')
cmap = plt.get_cmap("tab10")
for i in range(Nc):
    plotCamera(ax, R_est[i].T, - R_est[i].T @ t_est[i], cmap(i), 0.025)
    
# corners=np.array([[Y_est[0,0],Y_est[1,0],Y_est[2,0]],[Y_est[0,7],Y_est[1,7],Y_est[2,7]],[Y_est[0,47],Y_est[1,47],Y_est[2,47]],[Y_est[0,40],Y_est[1,40],Y_est[2,40]]])

# ax.plot(corners[:,0],corners[:,1],corners[:,2],'-')

fig.show()





with open('server_left_camera_calib_params.pkl', 'rb') as f:
    calib_params=pickle.load(f)
    rms,cameraMatrix,d,rvect,tvect=calib_params
    K[1]=cameraMatrix
    dist[1]=d

x[1,:,:]=x[2,:,:]

Rt_pairs = dict()
for i in range(Nc-1):
    for j in range(i+1, Nc):
        print(i,j)
        R, t, _, _, _ = excalib2(x[i], x[j], K[i], dist[i], K[j], dist[j])
        Rt_pairs[i, j] = np.hstack((R, t))

with open('stereo_left_camera_calib_params.pkl','rb') as f:
    calib_params=pickle.load(f)
    retval,K[0],dist[0],K[1],dist[1],R,T,E,F,P1,P2 = calib_params
    # Rt_pairs[0, 1][:,3]=T.reshape(-1)
    # Rt_pairs[0, 1][:,0:3] = R
    Rt_pairs[0, 1] = np.hstack((R, T))

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
for c in reversed(range(Nc)):
    t_est[c] /= np.linalg.norm(t_est[1])

# Projection matrix
P_est = []
for i in range(Nc):
    P_est.append(K[i] @ np.hstack((R_est[i], t_est[i])))
P_est = np.array(P_est)

# Triangulate 3D points
Y_est_2 = []
for i in range(Np):
    y = triangulate(x[0:Nc, i, :].reshape((-1, 2)), P_est)
    Y_est_2.append(y)
Y_est_2 = np.array(Y_est_2).T
Y_est_2 = Y_est_2[:3, :] / Y_est_2[3, :]

# ax.set_aspect('equal')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(0, 1)
ax.plot(Y_est_2[0, :], Y_est_2[1, :], Y_est_2[2, :], "o", label='C-L')
cmap = plt.get_cmap("tab10")
for i in range(Nc):
    plotCamera(ax, R_est[i].T, - R_est[i].T @ t_est[i], cmap(i), 0.025)
    
# corners=np.array([[Y_est[0,0],Y_est[1,0],Y_est[2,0]],[Y_est[0,7],Y_est[1,7],Y_est[2,7]],[Y_est[0,47],Y_est[1,47],Y_est[2,47]],[Y_est[0,40],Y_est[1,40],Y_est[2,40]]])

# ax.plot(corners[:,0],corners[:,1],corners[:,2],'-')

fig.show()

Y_est=[]
Y_est.append(Y_est_1)
Y_est.append(Y_est_2)
Y_est=np.array(Y_est)
Y_est_F=Y_est.mean(0)

ax.plot(Y_est_F[0, :], Y_est_F[1, :], Y_est_F[2, :], "o", label='Mean')

ax.legend()

print('ok')



from pycpd import RigidRegistration, AffineRegistration
# Generate random point clouds
source_points = Y_est_F[0:2,:].T
square_size = float(35)  # New larger calibration object

pattern_size = (8, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
target_points = pattern_points[:,0:2]

h, mask = cv2.findHomography(source_points,target_points,cv2.RANSAC)

reg = RigidRegistration(X=target_points, Y=source_points)
# run the registration & collect the results
TY, (s_reg, R_reg, t_reg) = reg.register()

fign=plt.figure()
plt.scatter(target_points[:, 0], target_points[:, 1], c='blue', label='Target Points')
plt.scatter(source_points[:, 0], source_points[:, 1], c='red', label='Source Points')
plt.scatter(TY[:, 0], TY[:, 1], c='red', label='Final')
# plt.legend()
plt.title('Reference Grid')
fign.show()
print('ok')


# def apply_transformation(points, scale, rotation, translation):
def apply_transformation(points,h):
    # Create homogeneous transformation matrix
    transformation_matrix = np.zeros((3, 3))
    # xscale=300.29581403300756
    # yscale=296.20262298152267
    # transformation_matrix[:2, :2] = scale * rotation
    # transformation_matrix[:2, 2] = translation
    # transformation_matrix=h
    transformation_matrix[2, 2] = 1.0
    
    transformation_matrix[0,0]=h[0,0]
    transformation_matrix[1,1]=h[1,1]
    # transformation_matrix[0,2]=h[0,2]
    # transformation_matrix[1,2]=h[1,2]
    

    # Convert points to homogeneous coordinates
    homogeneous_points = np.column_stack((points, np.ones(len(points))))

    # Apply transformation
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T

    # Extract the transformed points
    transformed_points = transformed_points[:, :2]

    return transformed_points

# myPoints=reg.transform_point_cloud(Y=Y_est[0:2,:].T)
myPoints=apply_transformation(Y_est_F[0:2,:].T,h)
fig5=plt.figure()
plt.scatter(myPoints[:,0],myPoints[:,1])
plt.title('Reconstructed Grid')
fig5.show()
print('ok')

error=np.abs(target_points-myPoints)
error=np.sqrt(error[:,0]**2+error[:,1]**2)
error=error.reshape((6,8))


fig6, ax = plt.subplots()
heatmap = ax.imshow(error, cmap='hot')
cbar = plt.colorbar(heatmap)
plt.title('Absolute Value of Error')
fig6.show()
print('ok')