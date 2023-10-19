# -*- coding: utf-8 -*-
"""

Camera calibration procedure for the postural camera system

"""

import os

# import posturalCam_master_NETWORK as backend
import posturalTemp as backend

folder = 'Calibration Videos Data/6'


def calibrate_client():
    """Calibrate the left camera"""
    proc = backend.posturalProc(v_fname=os.path.join(folder, 'client.h264'), kind='client')
    proc.get_calibration_frames()
    # proc.camera_calibration()


def calibrate_server_right():
    """Calibrate the left camera"""
    proc = backend.posturalProc(v_fname=os.path.join(folder, 'server_right.h264'), kind='server_right')
    proc.get_calibration_frames()
    # proc.camera_calibration()


def calibrate_server_left():
    """Calibrate the right camera"""
    proc = backend.posturalProc(v_fname=os.path.join(folder, 'server_left.h264'), kind='server_left')
    proc.get_calibration_frames()
    # proc.camera_calibration()


def stereo_left_calibrate():
    """Calibrate the stereo cameras extrinic parameters"""
    proc = backend.posturalProc(v_fname=os.path.join(folder, 'client.h264'), kind='client')
    proc2 = backend.posturalProc(v_fname=os.path.join(folder, 'server_left.h264'), kind='server_left')
    stereo = backend.stereo_process(proc, proc2, kind='left')

    # You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.get_calibration_frames()
    # stereo.stereo_calibrate()


def stereo_right_calibrate():
    """Calibrate the stereo cameras extrinic parameters"""
    proc = backend.posturalProc(v_fname=os.path.join(folder, 'client.h264'), kind='client')
    proc2 = backend.posturalProc(v_fname=os.path.join(folder, 'server_right.h264'), kind='server_right')
    stereo = backend.stereo_process(proc, proc2, kind='right')

    # You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.get_calibration_frames()
    # stereo.stereo_calibrate()


def stereo_left_right_calibrate():
    """Calibrate the stereo cameras extrinic parameters"""
    proc = backend.posturalProc(v_fname=os.path.join(folder, 'server_right.h264'), kind='server_right')
    proc2 = backend.posturalProc(v_fname=os.path.join(folder, 'server_left.h264'), kind='server_left')
    stereo = backend.stereo_process(proc, proc2, kind='left_right')

    # You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.get_calibration_frames()
    # stereo.stereo_calibrate()


if __name__ == '__main__':
    """Comment out the functions you dont want to call by placing a # before them.
    Any problems contact Oscar Giles o.t.giles@leeds.ac.uk"""

    calibrate_client()  # Calibnnnnnnnnnrate intrinsics of camera 1
    calibrate_server_right()
    calibrate_server_left()  # Calibrate intrinsics of camera 2

    stereo_left_calibrate()  # Calibrate stereo rig (camera extrinsics)
    stereo_right_calibrate()
    # stereo_left_right_calibrate()
