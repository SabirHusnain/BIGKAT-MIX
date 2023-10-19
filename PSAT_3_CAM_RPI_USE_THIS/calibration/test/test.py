import os, cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

master_dir = os.getcwd()

square_size = float(35)  # New larger calibration object

pattern_size = (8, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []

h, w = 0, 0

kind='serv_left'

fn=master_dir+f'/image/calib_img_{kind}_1.tiff'

print('processing %s... ' % fn, end='')
img = cv2.imread(fn, 0)

if img is None:
    print("Failed to load", fn)
    exit(0)

found, corners = cv2.findChessboardCorners(img, pattern_size)

if found:
    term = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    corners=cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
else:
    print('chessboard not found')
    exit(0)

img_points.append(corners.reshape(-1, 2))
obj_points.append(pattern_points)

try:
    with open(master_dir+'/point.pkl','rb') as f:
        x=pickle.load(f)
except:
    x=np.zeros((3,48,2))

x[2,:,:]=corners.reshape(-1, 2)

with open(master_dir+'/point.pkl','wb') as f:
    pickle.dump(x, f)

# plt.scatter(x[0,:,0],x[0,:,1])
# plt.show()

print('ok')