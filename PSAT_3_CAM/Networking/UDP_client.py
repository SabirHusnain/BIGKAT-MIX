# -*- coding: utf-8 -*-
"""

"""

import socket
import sys
import time

    
    
class UDP_client:
    
    def __init__(self, host, addr):
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = (host, addr) #The server's address
        
    def send_message(self, message):
        """Send a message to the server and wait for aknowledgment of reciept
        
        Message must be a byte stream"""
        
        sent = self.sock.sendto(message, self.server_address) #Try sending the time to the server
        
        data, server = self.sock.recvfrom(1024) #Wait for a response
        
        print('Message from Server: {}'.format(data.decode())) #Decode data and print it
        
    def close(self):
        
        self.sock.close()
        
myCli = UDP_client('localhost', 9999)


t = 3 #Time to record for
#message = 
#myCli.send_message(message)

myCli.send_message('{},{}'.format(time.time(), t).encode())
#myCli.close()