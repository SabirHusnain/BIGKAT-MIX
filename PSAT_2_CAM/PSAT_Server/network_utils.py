"""
Functions for sending and recieving messages over a TCP network connection.
Code by Adam Rosenfield on stack exchange
http://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data

Created and edited by Oscar Giles
o.t.giles@leeds.ac.uk
"""

import struct


def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>L', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
    
    
