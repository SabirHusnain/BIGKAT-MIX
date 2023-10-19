# -*- coding: utf-8 -*-
"""
"""
import threading, pickle
import socket
import sys
import numpy as np

def server():
    """Wait for a request to send over a video file. Then send it"""
    
    
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Create a socket object
    
    host = socket.gethostname() #Get local machine name
    port = 9999 #Choose a port
    
    print('TCP Server: starting on {} port {}'.format(host, port))
    
    #Bind to the port
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #Allow the address to be used again
    serversocket.bind((host, port)) 
    
    #Queue up 1 request    
    serversocket.listen(1)
  
        
    #Establist connection    
    clientsocket, addr = serversocket.accept()
    print("Server: Connection established with {}".format(addr))
    

    #Send over video file
   
    f_vid = open("test_file.h264", 'rb') #Open file to send
    
    while True:
        print("TCP Server: Sending file")
        l = f_vid.read(1024)
        if not l:
            break
        clientsocket.send(l)
        
    clientsocket.close()
        
    
    print("Server: Closing Socket")
    serversocket.close()
    print("Server: Socket closed")   

server()