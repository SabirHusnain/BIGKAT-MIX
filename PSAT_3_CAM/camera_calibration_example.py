# -*- coding: utf-8 -*-
"""

Camera calibration procedure for the postural camera system

"""

import os

# import posturalCam_master_NETWORK as backend
import posturalTemp as backend

folder = 'Calibration Videos Data/Calib 5'


def calibrate_client_left():
    """Calibrate the left camera"""
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveLeft.h264'), kind='client_left')
    proc.get_calibration_frames()
    proc.camera_calibration()


def calibrate_client_right():
    """Calibrate the left camera"""
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveRight.h264'), kind='client_right')
    proc.get_calibration_frames()
    proc.camera_calibration()


def calibrate_client_left_preview():
    """Calibrate the left camera"""
    proc = backend.posturalProc(kind='client_left_preview')

    proc.camera_calibration()


def calibrate_client_right_preview():
    """Calibrate the left camera"""
    proc = backend.posturalProc(kind='client_right_preview')

    proc.camera_calibration()


def calibrate_server():
    """Calibrate the right camera"""
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'Master.h264'), kind='server')
    proc.get_calibration_frames()
    proc.camera_calibration()


def calibrate_server_preview():
    """Calibrate the left camera"""
    proc = backend.posturalProc(kind='server_preview')

    proc.camera_calibration()


def stereo_left_calibrate():
    """Calibrate the stereo cameras extrinic parameters"""
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveLeft.h264'), kind='client_left')
    proc2 = backend.posturalProc(
        v_fname=os.path.join(folder, 'Master.h264'), kind='server')
    stereo = backend.stereo_process(proc, proc2, kind='left')

    # You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.get_calibration_frames()
    stereo.stereo_calibrate()


def stereo_right_calibrate():
    """Calibrate the stereo cameras extrinic parameters"""
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveRight.h264'), kind='client_right')
    proc2 = backend.posturalProc(
        v_fname=os.path.join(folder, 'Master.h264'), kind='server')
    stereo = backend.stereo_process(proc, proc2, kind='right')

    # You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.get_calibration_frames()
    stereo.stereo_calibrate()


def stereo_lef_right_calibrate():
    """Calibrate the stereo cameras extrinic parameters"""
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveRight.h264'), kind='client_right')
    proc2 = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveLeft.h264'), kind='server')
    stereo = backend.stereo_process(proc, proc2, kind='left_right')

    # You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.get_calibration_frames()
    stereo.stereo_calibrate()


def calibrate_stereo_left_preview():
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveLeft.h264'), kind='client_left_preview')
    proc2 = backend.posturalProc(
        v_fname=os.path.join(folder, 'Master.h264'), kind='server_preview')
    stereo = backend.stereo_process(proc, proc2, kind='left')

    #    stereo.get_calibration_frames() #You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.stereo_calibrate()

    return stereo


def calibrate_stereo_right_preview():
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveRight.h264'), kind='client_right_preview')
    proc2 = backend.posturalProc(
        v_fname=os.path.join(folder, 'Master.h264'), kind='server_preview')
    stereo = backend.stereo_process(proc, proc2, kind='right')

    #    stereo.get_calibration_frames() #You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.stereo_calibrate()

    return stereo


def calibrate_stereo_left_right_preview():
    proc = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveRight.h264'), kind='client_right_preview')
    proc2 = backend.posturalProc(
        v_fname=os.path.join(folder, 'slaveLeft.h264'), kind='server_preview')
    stereo = backend.stereo_process(proc, proc2, kind='left_right')

    #    stereo.get_calibration_frames() #You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.stereo_calibrate()

    return stereo


if __name__ == '__main__':
    """Comment out the functions you dont want to call by placing a # before them.
    Any problems contact Oscar Giles o.t.giles@leeds.ac.uk"""

    calibrate_client_left()  # Calibnnnnnnnnnrate intrinsics of camera 1
    calibrate_client_right()
    calibrate_server()  # Calibrate intrinsics of camera 2

    stereo_left_calibrate()  # Calibrate stereo rig (camera extrinsics)
    stereo_right_calibrate()
    stereo_lef_right_calibrate()

    # calibrate_client_left_preview() #Calibrate intrinsics of camera 1 (Preview mode)
    # calibrate_client_right_preview()
    # calibrate_server_preview()

    # s1 = calibrate_stereo_left_preview()
    # s2 = calibrate_stereo_right_preview
    # s3 = calibrate_stereo_left_right_preview()
