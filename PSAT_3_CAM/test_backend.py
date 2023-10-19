# -*- coding: utf-8 -*-
"""
"""

import ctypes
import os
import pdb
import sys
import threading
import time

import posturalCam_master_NETWORK as backend

backend_camera = backend.posturalCam()

backend_camera.TCP_client_start()

backend_camera.TCP_client_start_UDP(5, 'testIR.h264')  # Starts the video recording
backend_camera.TCP_client_request_timestamps()  # Request time stamps

backend_camera.TCP_client_request_video()  # Request video

#        IR_process_thread = threading.Thread(target = self.backend_camera.TCP_client_request_IRPoints)
#        IR_process_thread.start()

backend_camera.TCP_client_request_IRPoints()
