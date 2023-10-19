# -*- coding: utf-8 -*-
"""

"""

import socket
import sys, time

## Create a TCP/IP socket
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#
## Bind the socket to the port
#server_address = ('localhost', 10000)
#print('starting up on {} port {}'.format(server_address, server_address)) 

#sock.bind(server_address)
#
#while True:
#    print('waiting to receive message')
#    data, address = sock.recvfrom(4096)
#    data = data.decode()
#    delta = float(data) - time.time()
#    print(float(data) - time.time())
#    print('received {} bytes from {}'.format(len(data), address))
#    print('{}'.format(time.ctime(float(data))))
#    
#    if data:
#        sent = sock.sendto(str(delta).encode(), address)
#        print('sent {} bytes back to {}'.format(sent, address))
#        


class UDP_server:
    
    def __init__(self, host, addr):
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = (host, addr) #The server's address        
        self.sock.bind(self.server_address) #Bind the server
        
        self.get_messages() #Start getting messages
        
    def get_messages(self):
        """Query the time and send it to the server"""
        
        while True:
            print("Server: Waiting for messages")
            data, address =  self.sock.recvfrom(1024) #Wait for data
            data_decoded = data.decode() #Decode the data
            print("Server: {} bytes of data recieved from {}\n".format(len(data), address))
            print("Server: Data = {}".format(data_decoded))
            
            if data:
                sent = self.sock.sendto('recieved'.encode(), address)         
        
        
    def close(self):
        
        self.sock.close()
        
myServ = UDP_server('localhost', 10000)
