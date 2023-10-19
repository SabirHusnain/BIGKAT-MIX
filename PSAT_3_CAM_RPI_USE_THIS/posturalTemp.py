import sys
import time
import io
import os
import glob
import pickle
import socket
import threading
import subprocess
import multiprocessing
import psutil  # To monitor memory
import queue

# import paramikomultiprocessing

try:
    # Import picamera if we are not on windows (fudge to check if on RPi's)
    import picamera
    from picamera.array import PiRGBArray
except ImportError:
    print("ImportError: No module named 'picamera'")
import cv2
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
from pykalman import KalmanFilter  # For kalman filtering
from natsort import natsort
import pdb
import itertools
import struct
import network_utils as ntu


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


class posturalProc:
    """Play back video from the PiCamera with OpenCV and process the video files"""

    def __init__(self, cam_no=0, v_format='h264', v_fname="temp_out.h264", kind='client'):
        """Initialise the processing object. Pass the format and name of the video file to be analysed"""

        self.cam_no = cam_no
        self.v_format = v_format
        self.v_fname = v_fname
        self.v_loaded = False

        self.kind = kind
        self.resolution = (1280, 720)

        # If possible load the latest calibration parameters
        if self.kind == 'client':
            calib_fname = 'client_camera_calib_params.pkl'
        if self.kind == 'server_right':
            calib_fname = 'server_right_camera_calib_params.pkl'
        elif self.kind == 'server_left':
            calib_fname = 'server_left_camera_calib_params.pkl'

        # Or load the calibration for the preview cameras (which are a different resolution)
        elif self.kind == 'client_left_preview':
            calib_fname = 'client_left_preview_camera_calib_params.pkl'
        elif self.kind == 'client_right_preview':
            calib_fname = 'client_right_preview_camera_calib_params.pkl'
        elif self.kind == 'server_preview':
            calib_fname = 'server_preview_camera_calib_params.pkl'

        try:
            with open(os.path.join(os.getcwd(), "calibration", calib_fname), 'rb') as f:
                if sys.version_info[0] == 2:
                    # Pickle is different in python 3 vs 2
                    self.calib_params = pickle.load(f)
                else:
                    # Load the calib parameters. [ret, mtx, dist, rvecs, tvecs]
                    self.calib_params = pickle.load(f, encoding="Latin-1")
                self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs = self.calib_params
                print('{}: Initialisation: Camera calibration parameters were found'.format(
                    self.kind))
        except:
            print("{}:Initialisation: No camera calibration parameters were found".format(
                self.kind))

        if (not os.path.isdir('calibration/client/output')):
            os.makedirs('calibration/client/output')
        if (not os.path.isdir('calibration/server_right/output')):
            os.makedirs('calibration/server_right/output')
        if (not os.path.isdir('calibration/server_left/output')):
            os.makedirs('calibration/server_left/output')

    def load_video(self):
        """Load the video. The video filename was specified on initialisation"""

        if self.v_loaded == False:
            print("Loading video file: {}".format(self.v_fname))

            if self.v_fname in glob.glob('*.h264'):
                print("FILE EXISTS")
            self.cap = cv2.VideoCapture(self.v_fname)
            #            pdb.set_trace()
            if self.cap.isOpened() != True:
                raise IOError("Video could not be loaded. Check file exists")

            self.v_loaded = True
            print("Video Loading: Video Loaded from file: {}".format(self.v_fname))

        else:
            print("Video Loading: Video already loaded")

    def play_video(self, omx=True):
        """Play back the video. If omx == True and we are on the raspberry Pi it will play the video via the omxplayer"""

        if os.name != 'nt':
            RPi = True
        else:
            Rpi = False

        if not (Rpi and omx):
            self.load_video()
            cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

            while True:

                ret, frame = self.cap.read()

                if ret == False:
                    # If we are at the end of the video reload it and start again.
                    self.v_loaded = False
                    self.cap.release()  # Release the capture.
                    self.load_video()
                    continue  # Skip to next iteration of while loop

                cv2.imshow("video", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self.cap.release()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self.v_loaded = False

        elif Rpi:
            subprocess.call(['omx', self.v_fname])  # Call omx player

    def average_background(self):
        """Not working"""

        self.load_video()

        ret, frame = self.cap.read()
        avg1 = np.float32(frame)
        i = 0
        while True:
            print(i)
            i += 1
            ret, frame = self.cap.read()

            if ret == False:
                # If we are at the end of the video reload it and start again.
                break

            cv2.accumulateWeighted(frame, avg1, 0.05)
            res1 = cv2.convertScaleAbs(avg1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        plt.imshow(res1)

        self.cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        self.v_loaded = False

    def background_MOG(self):
        """Not working"""
        self.load_video()

        fgbg = cv2.createBackgroundSubtractorMOG2()

        while True:

            ret, frame = self.cap.read()

            if ret == False:
                # If we are at the end of the video reload it and start again.
                break

            fgmask = fgbg.apply(frame)

            thresh = cv2.threshold(fgmask, 2, 255, cv2.THRESH_BINARY)[1]

            #            frame[thresh == 0] = [0,0,0]
            #            plt.plot(fgmask.flatten())
            #            plt.show()
            #            pdb.set_trace()

            cv2.imshow("fgmask", fgmask)
            cv2.imshow("thresh", thresh)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def check_v_framerate(self, f_name="points.csv", ax=None):
        """Plot the camera framerate. Optionally pass the framerate file name"""

        frame_data = pd.read_csv(f_name)

        fps = 1000 / frame_data.diff()

        print(fps['0.0'].mean(), fps['0.0'].std())

        plt.plot(fps)
        plt.show()

    def get_calibration_frames(self):
        """Allows us to choose images for calibrating the RPi Camera. The more the merrier (get lots of images from different angles"""

        i = 0
        #        Check for previous files
        master_dir = os.getcwd()
        #        os.chdir(os.path.join(master_dir, "calibration", self.kind)) #Go to the calibration images directory
        file_list = glob.glob(os.path.join(
            master_dir, 'calibration', self.kind, '*.tiff'))
        output_list = glob.glob(os.path.join(
            master_dir, 'calibration', self.kind, 'output', '*'))

        if file_list != []:

            cont = input(
                "Calibration images exist. Press 'c' to overwrite or 'a' to append. Any other key to cancel ")

            if cont.lower() == 'c':

                for f in file_list:
                    os.remove(f)

                for f2 in output_list:
                    os.remove(f2)

            elif cont.lower() == 'a':

                # Gets the last image number
                last_num = natsort.humansorted(
                    file_list)[-1].split('.tiff')[0].split("_")[-1]
                last_num = int(last_num)

                i = last_num + 1

            else:
                print("Escaping calibration")
                return

        #        os.chdir(master_dir)

        print("Camera Calibration: Loading Video")
        self.load_video()  # Make sure the calibration video is loaded

        ret, frame = self.cap.read()

        while True:

            if ret == False:
                print("Camera Calibration: No more Frames")
                self.cap.release()
                cv2.destroyAllWindows()
                self.v_loaded = False
                break

            cv2.imshow("Calibration", frame)

            key_press = cv2.waitKey(1) & 0xFF

            if key_press == ord("n"):

                ret, frame = self.cap.read()
                continue

            elif key_press == ord("s"):
                print(ret)
                cv2.imwrite(os.path.join(os.getcwd(), 'calibration',
                                         self.kind, 'calib_img_{}.tiff'.format(i)), frame)
                #                np.save(os.path.join(os.getcwd(), 'calibration', 'calib_img_{}'.format(i)), frame)
                i += 1
                ret, frame = self.cap.read()

                continue

            elif key_press == ord("q"):
                self.cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                self.v_loaded = False
                break

    def camera_calibration(self):
        """Use the camera calibration images to try and calibrate the camera"""

        # Find all the images

        master_dir = os.getcwd()

        # Go to the calibration images directory
        os.chdir(os.path.join(master_dir, "calibration", self.kind))

        debug_dir = 'output'
        if not os.path.isdir(debug_dir):
            os.mkdir(debug_dir)

        img_names = glob.glob("*tiff")

        if len(img_names) == 0:
            img_names = glob.glob("*jpeg")

        square_size = float(35)  # New larger calibration object

        pattern_size = (8, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        self.obj_points = []
        self.img_points = []

        h, w = 0, 0
        img_names_undistort = []

        for fn in img_names:
            print('processing %s... ' % fn, end='')
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Failed to load", fn)
                continue

            h, w = img.shape[:2]

            found, corners = cv2.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
                corners=cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            if debug_dir:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                path, name, ext = splitfn(fn)

                outfile = os.path.join(debug_dir, name + '_chess.png')
                cv2.imwrite(outfile, vis)
                if found:
                    img_names_undistort.append(outfile)

            if not found:
                print('chessboard not found')
                continue
                
            self.img_points.append(corners.reshape(-1, 2))        
            self.obj_points.append(pattern_points)

            print('ok')

        # calculate camera distortion
        self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points,
                                                                                                    self.img_points,
                                                                                                    (w, h), None, None)
        #print("camera matrix:\n", self.camera_matrix)
        #self.camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h))

        print("\nRMS:", self.rms)
        print("camera matrix:\n", self.camera_matrix)
        print("distortion coefficients: ", self.dist_coefs)

        # Save distortion parameters to file
        if self.kind == 'server_right':
            fname = 'server_right_camera_calib_params.pkl'

        elif self.kind == 'client':
            fname = 'client_camera_calib_params.pkl'

        elif self.kind == 'server_left':
            fname = 'server_left_camera_calib_params.pkl'

        calib_params = self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs

        if self.kind == 'server_right':
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'server_right_camera_calib_params.pkl')

        elif self.kind == 'client':
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'client_camera_calib_params.pkl')

        elif self.kind == 'server_left':
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'server_left_camera_calib_params.pkl')

        elif self.kind == 'server_preview':
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'server_preview_camera_calib_params.pkl')

        elif self.kind == 'client_left_preview':
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'client_left_preview_camera_calib_params.pkl')

        elif self.kind == 'client_right_preview':
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'client_right_preview_camera_calib_params.pkl')

        with open(fname, 'wb') as f:
            # THIS MAY BE BETTER AS A TEXT FILE. NO IDEA IF THIS WILL LOAD ON THE RASPBERRY PI BECAUSE IT IS A BINARY FILE
            pickle.dump(calib_params, f)

        # undistort the image with the calibration
        print('')
        for img_found in img_names_undistort:
            img = cv2.imread(img_found)

            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h))

            dst = cv2.undistort(img, self.camera_matrix,
                                self.dist_coefs, None, newcameramtx)

            # crop and save the image
            #            x, y, w, h = roi
            #            dst = dst[y:y+h, x:x+w]
            outfile = img_found.split(".png")[0] + '_undistorted.png'
            print('Undistorted image written to: %s' % outfile)
            cv2.imwrite(outfile, dst)

        cv2.destroyAllWindows()

        os.chdir(master_dir)

    def camera_calibration2(self):
        """DEPRECIATED
        Run after self.camera_calibration. This does the actual optimisizing and saves the parameters to file"""

        master_dir = os.getcwd()
        # Go to the calibration images directory
        os.chdir(os.path.join(master_dir, "calibration"))
        # Run the calibration
        mtx = np.array([[400, 0, self.resolution[0] / 2],
                        [0, 400, self.resolution[1] / 2],
                        [0, 0, 1]])
        print(mtx)
        print("Camera Calibration: Calibrating Camera. Please wait... This may take several minutes or longer")
        self.calib_params = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.resolution, mtx, None,
                                                flags=cv2.CALIB_ZERO_TANGENT_DIST)

        print(self.calib_params[1])
        #        self.calib_params = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.resolution, proc.calib_params[1], proc.calib_params[2],
        #                                                flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST)

        self.calib_params = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.resolution, self.calib_params[1],
                                                self.calib_params[2],
                                                flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        print("Reprojection Error: {}".format(self.calib_params[0]))
        self.calib_params = list(self.calib_params)

        self.calib_params.append(self.objpoints)
        self.calib_params.append(self.imgpoints)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs, self.objpoints, self.imgpoints = self.calib_params
        print("Camera Calibration: Calibration complete")

        if self.kind == 'server':
            fname = 'server_camera_calib_params.pkl'

        elif self.kind == 'client_left':
            fname = 'client_left_camera_calib_params.pkl'

        elif self.kind == 'client_right':
            fname = 'client_right_camera_calib_params.pkl'

        with open(fname, 'wb') as f:
            pickle.dump(self.calib_params, f)

        os.chdir(master_dir)  # Set directory back to master directory.

    def camera_calibration_circle(self):
        """DEPRECIATED
        Use the camera calibration images to try and calibrate the camera"""

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0)... (8,5,0)

        squares = (11, 4)

        #        ##Square number format
        self.objp = np.zeros((np.prod(squares), 3), np.float32)
        objp[:, :2] = np.mgrid[0:squares[0], 0:squares[1]].T.reshape(-1, 2)
        #
        # mm format
        square_size = 24.5  # mm

        #        objp = np.zeros((np.prod(squares), 3), np.float32)

        #        p = itertools.product(np.arange(0,square_size*squares[0],square_size), np.arange(0,square_size*squares[1],square_size))
        #        objp[:,:2] = np.array([i for i in p])[:,::-1]

        # Arrays to store object points and image points for all the images
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane

        master_dir = os.getcwd()
        # Go to the calibration images directory
        os.chdir(os.path.join(master_dir, "calibration"))

        images = glob.glob("*npy")

        len_images = len(images)
        i = 0
        for fname in images:

            img = np.load(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #            gray
            #            plt.imshow(gray, cmap = 'gray')
            #            plt.show()
            #            pdb.set_trace()
            ret, corners = cv2.findCirclesGrid(
                gray, squares, None, cv2.CALIB_CB_ASYMMETRIC_GRID)

            if ret == True:
                print("Camera Calibration: Processing Image: {} of {}".format(
                    i, len_images))
                self.objpoints.append(objp)
                #                cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

                self.imgpoints.append(corners)

                cv2.drawChessboardCorners(img, squares, corners, ret)
                cv2.imshow("img", img)
                cv2.waitKey(250)

            i += 1
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    #        pdb.set_trace()
    # Run the calibration

    #        print("Camera Calibration: Calibrating Camera. Please wait... This may take several minutes or longer")
    #        self.calib_params = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #        print("Reprojection Error: {}".format(self.calib_params[0]))
    #        self.calib_params = list(self.calib_params)
    #        self.calib_params.append(objpoints)
    #        self.calib_params.append(imgpoints)
    #        print("Camera Calibration: Calibration complete")
    #
    #        if self.kind == 'server':
    #            fname = 'server_camera_calib_params.pkl'
    #
    #        elif self.kind == 'client':
    #            fname = 'client_camera_calib_params.pkl'
    #
    #        with open(fname, 'wb') as f:
    #            pickle.dump(self.calib_params, f)
    #
    #        os.chdir(master_dir) #Set directory back to master directory.

    def check_camera_calibration(self):
        """DEPRECIATED
        Check the reprojection error"""

        tot_error = 0

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(
                self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error

        print("Check Camera Calibration: Total Error: {}".format(
            tot_error / len(self.objpoints)))

    def undistort(self, img):
        """undistort an image"""
        h, w = img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)

        return dst

    def undistort_points(self, p):
        """Undistort points p where p is a 1XNx2 array or an NX1X2 array"""

        dst = cv2.undistortPoints(
            p, self.camera_matrix, self.dist_coefs, P=self.camera_matrix)

        return dst

    def cube_render(self, img=None):
        """DEPRECIATED
        Render a cube over an image

        DOESNT WORK AT THE MOMENT FOR SOME REASON! WONT SHOW IMAGE"""

        def draw(img, corners, imgpts):

            corner = tuple(corners[0].ravel())

            img = cv2.line(img, corner, tuple(
                imgpts[0].ravel()), (255, 0, 0), 5)
            img = cv2.line(img, corner, tuple(
                imgpts[1].ravel()), (0, 255, 0), 5)
            img = cv2.line(img, corner, tuple(
                imgpts[2].ravel()), (0, 0, 255), 5)

            return img

        def draw_cube(img, corners, imgpts):

            imgpts = np.int32(imgpts).reshape(-1, 2)

            img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

            for i, j in zip(range(4), range(4, 8)):
                img = cv2.line(img, tuple(
                    imgpts[i]), tuple(imgpts[j]), (255), 3)

            img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

            return img

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0)... (8,5,0)
        squares = (9, 6)
        objp = np.zeros((np.prod(squares), 3), np.float32)
        objp[:, :2] = np.mgrid[0:squares[0], 0:squares[1]].T.reshape(-1, 2)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

        axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                           [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

        if img == None:

            self.load_video()
            i = 0
            while True:
                print("Render Cube: Frame {}".format(i))
                i += 1
                ret, frame = self.cap.read()

                if ret == False:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret2, corners = cv2.findChessboardCorners(gray, squares, None)

                if ret2:
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)

                    retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(
                        np.expand_dims(objp, axis=1), corners2, self.mtx, self.dist)

                    imgpts, jac = cv2.projectPoints(
                        axis, rvecs, tvecs, self.mtx, self.dist)

                    frame = draw_cube(frame, corners2, imgpts)  # Renders cube

                cv2.imshow('show', frame)
                cv2.waitKey(1)

        self.cap.release()

        cv2.destroyAllWindows()

    def get_markers_blob(self):

        self.load_video()

        i = 0
        next_frame = True

        #        cv2.namedWindow("TB")
        #        cv2.createTrackbar("minThresh", "TB", 0,255, nothing)
        #        cv2.createTrackbar("maxThresh", "TB", 0,255, nothing)
        #
        #        cv2.createTrackbar("minArea", "TB", 0,255, nothing)
        #        cv2.createTrackbar("maxArea", "TB", 500,1500, nothing)
        #        cv2.createTrackbar("minCircularity", "TB", 8, 10, nothing)
        #
        #        cv2.createTrackbar("minConvex", "TB", 87,255, nothing)
        #        cv2.createTrackbar("minIntRatio", "TB", 5,255, nothing)

        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255

        params.filterByColor = True
        params.blobColor = 255

        params.filterByArea = True
        params.minArea = 0
        params.maxArea = 1500

        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.1

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)

        while True:

            print(i)

            if next_frame:
                next_frame = False
                ret, frame = self.cap.read()

                if ret == False:
                    break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #           cv2.imshow("gray", gray)

            keypoints = detector.detect(gray)
            im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array(
                []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #           cv2.imshow("Raw", frame)
            cv2.imshow("blobs", im_with_keypoints)
            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord("n"):
                break
            next_frame = True
            i += 1
            if key_press == ord("p"):
                next_frame = True
                i += 1

    def get_markers(self):
        """Processs the video to get the IR markers
        Currently tracking a mobile phone light. May need to get a visible light filter to only allow IR light through"""

        marker_file = "marker_data.csv"
        #        f = io.open(marker_file, 'w')
        self.load_video()

        cv2.namedWindow("HSV TRACKBARS")
        cv2.createTrackbar("Hue low", "HSV TRACKBARS", 0, 255, nothing)
        cv2.createTrackbar("Hue high", "HSV TRACKBARS", 255, 255, nothing)
        cv2.createTrackbar("Sat low", "HSV TRACKBARS", 0, 255, nothing)
        cv2.createTrackbar("Sat high", "HSV TRACKBARS", 255, 255, nothing)
        cv2.createTrackbar("Val low", "HSV TRACKBARS", 0, 255, nothing)
        cv2.createTrackbar("Val high", "HSV TRACKBARS", 255, 255, nothing)
        cv2.createTrackbar("Radius Low", "HSV TRACKBARS", 0, 50, nothing)
        cv2.createTrackbar("Radius High", "HSV TRACKBARS", 50, 500, nothing)

        i = 0
        next_frame = True

        while True:

            print(i)

            if next_frame:
                next_frame = False
                ret, frame = self.cap.read()

                if ret == False:
                    break

            frame2 = frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            hue_low = cv2.getTrackbarPos("Hue low", "HSV TRACKBARS")
            hue_high = cv2.getTrackbarPos("Hue high", "HSV TRACKBARS")
            sat_low = cv2.getTrackbarPos("Sat low", "HSV TRACKBARS")
            sat_high = cv2.getTrackbarPos("Sat high", "HSV TRACKBARS")
            val_low = cv2.getTrackbarPos("Val low", "HSV TRACKBARS")
            val_high = cv2.getTrackbarPos("Val high", "HSV TRACKBARS")
            r_low = cv2.getTrackbarPos("Radius Low", "HSV TRACKBARS")
            r_high = cv2.getTrackbarPos("Radius High", "HSV TRACKBARS")

            lower_blue = np.array([hue_low, sat_low, val_low])
            upper_blue = np.array([hue_high, sat_high, val_high])

            #           lower_blue = np.array([0,0,255])
            #           upper_blue = np.array([255,255,255])

            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            cnts = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None

            if len(cnts) > 0:

                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                if r_low < radius and radius < r_high:

                    cv2.circle(frame2, (int(x), int(y)),
                               int(radius), (0, 0, 255), 2)
                    cv2.circle(frame2, (int(x), int(y)), 1, (255, 0, 255), 1)

                    cv2.circle(mask, (int(x), int(y)),
                               int(radius), (0, 0, 255), 2)
                    cv2.circle(mask, (int(x), int(y)), 1, (255, 0, 255), 1)

                #                   f.write("{},{}\n".format(x,y))

                else:
                    #                   f.write("{},{}\n".format(-9999, -9999))
                    pass

            cv2.imshow("Raw", frame2)
            cv2.imshow("Mask", mask)

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord("n"):
                break

            if key_press == ord("p"):
                next_frame = True
                i += 1

        f.close()

    def ir_marker(self, img):
        """A function to get a single IR marker from and images. Returns the first marker it finds (So terrible if other IR light sources in)"""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        t_val = 100
        t, thresh = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:

            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            #                   if r_low < radius and radius < r_high:

            return (x, y), radius, center

        else:
            return None  # Return NaN if no marker found

    def get_ir_markers(self, out_queue=None, plot=False):
        """Processs the video to get the IR markers

        Arguments:

            plot: Save plot images to the marker folder
        """

        if plot == True:

            plot_loc = self.kind
            print(self.kind)

        else:
            plot_loc = None

        self.load_video()

        all_markers = []

        i = 0

        while self.cap.isOpened():

            ret, frame = self.cap.read()

            if not ret:
                self.cap.release()
                break

            ir = ir_marker.find_ir_markers(
                frame, n_markers=2, it=i, tval=150, plot_loc=plot_loc)

            all_markers.append(ir)  # Put results back into order

            i += 1
            print("Frame: {}".format(i))

        if out_queue != None:
            out_queue.put(all_markers)

        return all_markers

    def get_ir_markers_parallel(self, out_queue=None):
        """"Get IR markers from a video file. 
        out_queue is an optionally queue argument. If included the data will be put into the queue


        THERE IS A BUG HERE. THE WORKERS WILL NOT JOIN!!! 
        """

        self.load_video()  # Load the video

        # Create the Workers

        # Maximum number of items that can go in queue (may have to be small on RPi)
        max_jobs = 5
        jobs = multiprocessing.Queue(max_jobs)
        results = multiprocessing.Queue()  # Queue to place the results into

        n_workers = multiprocessing.cpu_count()

        workers = []

        for i in range(n_workers):
            print("Starting Worker {}".format(i))
            tmp = multiprocessing.Process(
                target=ir_marker.get_ir_marker_process, args=(jobs, results,))
            tmp.start()
            workers.append(tmp)

            print("Worker started")

        print("There are {} workers".format(len(workers)))
        i = 0
        while self.cap.isOpened():

            if not jobs.full():
                ret, frame = self.cap.read()

                if not ret:
                    self.cap.release()
                    break
                ID = i
                # Put a job in the Queue (the frame is the job)
                jobs.put([ID, frame])
                if i % 10 == 0:
                    print("Get Markers Parallel: Job {} put in Queue".format(i))

                i += 1

        print("Total jobs = {}".format(i))
        print("All jobs complete")
        N_jobs = i
        # Tell all workers to Die
        for worker in workers:
            print("KILLING MESSAGE")
            jobs.put("KILL")

        # Get everything out the results queue into a list
        output = []
        N_jobs_returned = 0
        print("HERE")
        print(N_jobs)

        while (N_jobs_returned != N_jobs):
            dat = results.get()
            output.append(dat)
            N_jobs_returned += 1
            print("RETURNED: {}, Total: {}".format(N_jobs_returned, N_jobs))

        print("KILL MY WORKERS")

        # Wait for workers to Die
        for worker in workers:
            worker.join()

            print("Worker joined")

        self.v_loaded = False

        output.sort()

        if not np.alltrue(np.diff(np.array([i[0] for i in output])) == 1):
            raise Exception("ORDER PROBLEM")

        # If a queue object exists put it into it. Useful if function is passed as a thread
        if out_queue != None:
            out_queue.put(output)

        all_markers = [m[1] for m in output]  # Get rid of marker ordering
        print("FINISHED MARKER PROCESSING")
        return all_markers


class stereo_process:

    def __init__(self, cam_client, cam_serv, kind):
        """Pass two camera processing objects"""

        self.cam_serv = cam_serv
        self.cam_client = cam_client
        self.kind = kind

        self.resolution = self.cam_client.resolution

        self.rectif = None  # None if stereorectify has not been called

        if (self.kind == 'left'):
            pkl_file = 'stereo_left_camera_calib_params.pkl'
        elif (self.kind == 'right'):
            pkl_file = 'stereo_right_camera_calib_params.pkl'
        elif (self.kind == 'left_right'):
            pkl_file = 'stereo_left_right_camera_calib_params.pkl'

        try:
            with open(os.path.join(os.getcwd(), "calibration", pkl_file), 'rb') as f:

                if sys.version_info[0] == 2:
                    # Pickle is different in python 3 vs 2
                    self.stereo_calib_params = pickle.load(f)

                else:
                    # Load the calib parameters. [ret, mtx, dist, rvecs, tvecs]
                    self.stereo_calib_params = pickle.load(
                        f, encoding="Latin-1")

                self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F, self.P1, self.P2 = self.stereo_calib_params
                print('Initialisation: Stereo Camera calibration parameters were found')
        except:
            print("Initialisation: No stereo camera calibration parameters were found")

        if (not os.path.isdir('calibration/stereo')):
            os.mkdir('calibration/stereo')
        if (not os.path.isdir('calibration/stereo/client_left')):
            os.mkdir('calibration/stereo/client_left')
        if (not os.path.isdir('calibration/stereo/client_right')):
            os.mkdir('calibration/stereo/client_right')
        if (not os.path.isdir('calibration/stereo/server_left')):
            os.mkdir('calibration/stereo/server_left')
        if (not os.path.isdir('calibration/stereo/server_right')):
            os.mkdir('calibration/stereo/server_right')
        if (not os.path.isdir('calibration/stereo/server_stereo_left')):
            os.mkdir('calibration/stereo/server_stereo_left')
        if (not os.path.isdir('calibration/stereo/server_stereo_right')):
            os.mkdir('calibration/stereo/server_stereo_right')

    def get_calibration_frames(self):
        """Allows us to choose images for calibrating the RPi Camera. The more the merrier (get lots of images from different angles"""

        resolution = (1280, 720)  # Image resolution
        # Check for previous files

        i = 0
        # Check for previous files
        master_dir = os.getcwd()

        #        os.chdir(os.path.join(master_dir, "calibration", 'stereo')) #Go to the calibration images directory
        if (self.kind == 'left'):
            client_file_list = glob.glob(os.path.join(
                master_dir, "calibration", 'stereo', 'client_left', 'calib*'))
            server_file_list = glob.glob(os.path.join(
                master_dir, "calibration", 'stereo', 'server_left', 'calib*'))
        elif (self.kind == 'right'):
            client_file_list = glob.glob(os.path.join(
                master_dir, "calibration", 'stereo', 'client_right', 'calib*'))
            server_file_list = glob.glob(os.path.join(
                master_dir, "calibration", 'stereo', 'server_right', 'calib*'))
        elif (self.kind == 'left_right'):
            client_file_list = glob.glob(os.path.join(
                master_dir, "calibration", 'stereo', 'server_stereo_right', 'calib*'))
            server_file_list = glob.glob(os.path.join(
                master_dir, "calibration", 'stereo', 'server_stereo_left', 'calib*'))

        if (client_file_list != []) and (server_file_list != []):

            cont = input(
                "Calibration images exist. Press 'c' to overwrite or 'a' to append. Any other key to cancel ")

            if cont.lower() == 'c':
                for f in client_file_list:
                    os.remove(f)

                for f in server_file_list:
                    os.remove(f)

            elif cont.lower() == 'a':
                # Gets the last image number
                last_num = natsort.humansorted(
                    client_file_list)[-1].split('.tiff')[0].split("_")[-1]
                last_num = int(last_num)
                i = last_num + 1

            else:
                print("Escaping calibration")
                return

        print("Camera Calibration: Loading Video")

        # Load both videos
        self.cam_serv.load_video()  # Make sure the calibration video is loaded
        self.cam_client.load_video()

        # Get first frame from both videos
        ret_serv, frame_serv = self.cam_serv.cap.read()
        ret_client, frame_client = self.cam_client.cap.read()

        cv2.namedWindow("Server", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Client", cv2.WINDOW_NORMAL)
        #               cv2.resizeWindow("Raw", int(resolution[0]/2), int(resolution[1]/2))
        while True:

            if ret_serv == False or ret_client == False:
                print("Camera Calibration: No more Frames")
                self.cam_serv.cap.release()
                self.cam_client.cap.release()
                cv2.destroyAllWindows()

                self.cam_serv.v_loaded = False
                self.cam_client.v_loaded = False
                break

            cv2.imshow("Server", frame_serv)
            cv2.imshow("Client", frame_client)

            cv2.resizeWindow("Server", int(
                resolution[0] / 2), int(resolution[1] / 2))
            cv2.resizeWindow("Client", int(
                resolution[0] / 2), int(resolution[1] / 2))

            key_press = cv2.waitKey(1) & 0xFF

            if key_press == ord("n"):

                ret_serv, frame_serv = self.cam_serv.cap.read()
                ret_client, frame_client = self.cam_client.cap.read()

                continue

            elif key_press == ord("s"):

                if (self.kind == 'left'):
                    cv2.imwrite(os.path.join(os.getcwd(), 'calibration', 'stereo',
                                             'server_left', 'calib_img_serv_{}.tiff'.format(i)), frame_serv)
                    cv2.imwrite(os.path.join(os.getcwd(), 'calibration', 'stereo',
                                             'client_left', 'calib_img_client_{}.tiff'.format(i)), frame_client)
                elif (self.kind == 'right'):
                    cv2.imwrite(os.path.join(os.getcwd(), 'calibration', 'stereo',
                                             'server_right', 'calib_img_serv_{}.tiff'.format(i)), frame_serv)
                    cv2.imwrite(os.path.join(os.getcwd(), 'calibration', 'stereo',
                                             'client_right', 'calib_img_client_{}.tiff'.format(i)), frame_client)
                elif (self.kind == 'left_right'):
                    cv2.imwrite(os.path.join(os.getcwd(), 'calibration', 'stereo',
                                             'server_stereo_left', 'calib_img_serv_{}.tiff'.format(i)), frame_serv)
                    cv2.imwrite(os.path.join(os.getcwd(), 'calibration', 'stereo',
                                             'server_stereo_right', 'calib_img_client_{}.tiff'.format(i)), frame_client)

                #                np.save(os.path.join(os.getcwd(), 'calibration', 'calib_img_serv_{}'.format(i)), frame_serv)
                #                np.save(os.path.join(os.getcwd(), 'calibration', 'calib_img_client_{}'.format(i)), frame_client)

                i += 1
                ret_serv, frame_serv = self.cam_serv.cap.read()
                ret_client, frame_client = self.cam_client.cap.read()
                continue

            elif key_press == ord("q"):
                self.cam_serv.cap.release()
                self.cam_client.cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                self.cam_serv.v_loaded = False
                self.cam_client.v_loaded = False
                break

    def stereo_calibrate(self):

        print("Running Stereo Calibration")
        master_dir = os.getcwd()

        # Go to the calibration images
        os.chdir(os.path.join(master_dir, "calibration", 'stereo'))

        if (self.kind == 'left'):
            client_dir = 'client_left'
            server_dir = 'server_left'
        elif (self.kind == 'right'):
            client_dir = 'client_right'
            server_dir = 'server_right'
        elif (self.kind == 'left_right'):
            client_dir = 'server_stereo_left'
            server_dir = 'server_stereo_right'

        img_names_client = glob.glob("{}/*tiff".format(client_dir))
        img_names_server = glob.glob("{}/*tiff".format(server_dir))

        if (len(img_names_client) == 0) and (len(img_names_server) == 0):
            img_names_client = glob.glob("{}/*jpeg".format(client_dir))
            img_names_server = glob.glob("{}/*jpeg".format(server_dir))

        img_names_client = natsort.humansorted(img_names_client)
        img_names_server = natsort.humansorted(img_names_server)

        square_size = float(35)

        pattern_size = (8, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        self.obj_points = []
        self.img_points_client = []
        self.img_points_server = []
        h, w = 0, 0

        for fn in range(len(img_names_client)):
            print('processing %s... ' % img_names_client[fn], end='')
            print('processing %s... ' % img_names_server[fn], end='')

            img_client = cv2.imread(img_names_client[fn], 0)
            img_server = cv2.imread(img_names_server[fn], 0)

            if img_client is None:
                print("Failed to load", img_names_client[fn])
                continue

            if img_server is None:
                print("Failed to load", img_names_server[fn])
                continue

            h, w = img_client.shape[:2]

            found_client, corners_client = cv2.findChessboardCorners(
                img_client, pattern_size)
            found_server, corners_server = cv2.findChessboardCorners(
                img_server, pattern_size)

            if found_client and found_server:
                term = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
                corners_client=cv2.cornerSubPix(img_client, corners_client,
                                                      (5, 5), (-1, -1), term)
                corners_server=cv2.cornerSubPix(img_server, corners_server,
                                                      (5, 5), (-1, -1), term)
                                                      
                self.img_points_client.append(corners_client.reshape(-1,2))
                self.img_points_server.append(corners_server.reshape(-1,2))
                self.obj_points.append(pattern_points)

            if not (found_client or found_server):
                print('chessboard not found in both images')
                continue

            print('ok')

        self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.obj_points, self.img_points_client, self.img_points_server,
            self.cam_client.camera_matrix, self.cam_client.dist_coefs,
            self.cam_serv.camera_matrix, self.cam_serv.dist_coefs, (w, h), criteria=term, flags=cv2.CALIB_FIX_INTRINSIC)

        self.P1 = np.dot(self.cameraMatrix1, np.hstack((np.identity(3),np.zeros((3,1)))))  #Projection Matrix for client cam
        self.P2 = np.dot(self.cameraMatrix2, np.hstack((self.R,self.T))) #Projection matrix for server cam
        print("\nRMS:", self.retval)

        self.stereo_calib_params = self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F, self.P1, self.P2

        if (self.kind == 'left'):
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'stereo_left_camera_calib_params.pkl')
        elif (self.kind == 'right'):
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'stereo_right_camera_calib_params.pkl')
        elif (self.kind == 'left_right'):
            fname = os.path.join(os.path.dirname(
                os.getcwd()), 'stereo_left_right_camera_calib_params.pkl')

        with open(fname, 'wb') as f:
            pickle.dump(self.stereo_calib_params, f)

        os.chdir(master_dir)

    def stereo_rectify(self):
        """
        DEPRECIATED
        This function only works when the image has resolution (1280, 720) """

        #        self.T[0,0] = -100
        self.rectif = cv2.stereoRectify(self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2,
                                        (1280, 720),
                                        self.R, self.T,
                                        flags=cv2.CALIB_ZERO_DISPARITY,
                                        alpha=1, newImageSize=(0, 0))

        self.R1, self.R2, self.P1, self.P2, self.Q, _o, _oo = self.rectif

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.cameraMatrix1, self.distCoeffs1, self.R1, self.P1, (1280, 720), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.cameraMatrix2, self.distCoeffs2, self.R2, self.P2, (1280, 720), cv2.CV_32FC1)

    def triangulate(self, points1, points2):
        """NOT WORKING YET"""

        z = cv2.triangulatePoints(self.P1, self.P2, points1, points2)

        z = (z / z[-1]).T
        z = z[:, :3]
        return z

    def test_triangulate(self, img1, img2):

        square_size = 24.5

        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        self.obj_points = []
        self.img_points_client = []
        self.img_points_server = []
        h, w = 0, 0

        img_client = img1
        img_server = img2

        h, w = img_client.shape[:2]

        found_client, corners_client = cv2.findChessboardCorners(
            img_client, pattern_size)
        found_server, corners_server = cv2.findChessboardCorners(
            img_server, pattern_size)

        if found_client and found_server:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img_client, corners_client,
                             (5, 5), (-1, -1), term)
            cv2.cornerSubPix(img_server, corners_server,
                             (5, 5), (-1, -1), term)

        if not (found_client and found_server):
            print('chessboard not found in both images')
            return None

        print('ok')
        return corners_client, corners_server

    def triangulate_all_get_PL(self, client_markers, server_markers):
        """Get points from both videos and triangulate to 3d. Return path length measure.
        I should break this into smaller functions.
        This function is for code testing and will be very slow on the RPi. Use on desktop


        Update on 07/09/16
        Recoding to allow for multiple marker triangulation
        Points always have to be passed        
        """

        client_n_markers, server_n_markers = client_markers.shape[1], server_markers.shape[1]

        # Make sure we have the correct number of markers in each camera
        assert client_n_markers == server_n_markers

        # Force the markers to have the same number of frames by truncating the end of which ever is the longer marker array
        pos1_len = client_markers.shape[0]
        pos2_len = server_markers.shape[0]
        client_markers = client_markers[:min(pos1_len, pos2_len)]
        server_markers = server_markers[:min(pos1_len, pos2_len)]

        # AN NxMx3 array of 3d marker positions
        markers_3d_all = np.empty(
            (client_markers.shape[0], client_markers.shape[1], 3))

        # Triangulate each marker point
        for mark in range(client_n_markers):
            pos1 = client_markers[:, mark]
            pos2 = server_markers[:, mark]

            #            print(pos1.shape)
            pos1_undistort = self.cam_client.undistort_points(
                np.expand_dims(pos1, 0)).squeeze(axis=0)
            pos2_undistort = self.cam_serv.undistort_points(
                np.expand_dims(pos2, 0)).squeeze(axis=0)

            pos1_corrected, pos2_corrected = cv2.correctMatches(
                self.F, np.expand_dims(pos1_undistort, 0), np.expand_dims(pos2_undistort, 0))

            pos1_corrected, pos2_corrected = pos1_corrected.squeeze(), pos2_corrected.squeeze()

            #            pos3d = self.triangulate(pos1_undistort.T, pos2_undistort.T)
            pos3d = self.triangulate(pos1_corrected.T, pos2_corrected.T)

            markers_3d_all[:, mark, :] = pos3d

        return markers_3d_all

    def kalman_smoother(self, markers_3d):
        """Filter the 3d marker points with a kalman smoother

        Must be an NxMX3 array where N is the number of data points and M is the number of markers. The last axis is the x,y,x position of the marker"""

        n_markers = markers_3d.shape[1]

        filtered_markers = np.empty(markers_3d.shape)

        for mark in range(n_markers):
            pos3d = markers_3d[:, mark]
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=3)

            dt = 1 / 60
            transition_M = np.array([[1, 0, 0, dt, 0, 0],
                                     [0, 1, 0, 0, dt, 0],
                                     [0, 0, 1, 0, 0, dt],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]])

            observation_M = np.array([[1, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0]])

            measurements = np.ma.masked_invalid(pos3d)

            initcovariance = 1000 * np.eye(6)
            transistionCov = 0.5 * np.eye(6)
            observationCov = 3 * np.eye(3)

            kf = KalmanFilter(transition_matrices=transition_M, observation_matrices=observation_M,
                              initial_state_covariance=initcovariance, transition_covariance=transistionCov,
                              observation_covariance=observationCov)

            (filtered_state_means, filtered_state_covariances) = kf.smooth(measurements)

            # Don't return state covariance for a while
            filtered_markers[:, mark] = filtered_state_means[:, :3]

        return filtered_markers
