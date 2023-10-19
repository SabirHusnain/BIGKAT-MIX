# -*- coding: utf-8 -*-
"""
"""
import threading, pickle
import socket
import sys
import numpy as np


def client():
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    host = socket.gethostname()
    port = 9999
    
    s.connect((host, port))
    
    f_video_recv = open("recieved_vid.h264", 'wb')

    while True:
        data = s.recv(1024)
        
        if not data:
            break
        
        f_video_recv.write(data)    
        
    s.close()
    

    
#    pickle.loads(tm)

client()