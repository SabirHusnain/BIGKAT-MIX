# -*- coding: utf-8 -*-
"""
A simple script to connect to a raspberry pi and ask for it's IP address using paramiko
"""

import paramiko


class shh_client:
    def __init__(self,ip,userame,password):
        self.ip=ip
        self.username=username
        self.password=password
        self.connect()  # Try to connect to RPi Server

    def connect(self):
        try:
            self.shh = paramiko.SSHClient()
            self.shh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            addr = self.ip
            username = self.username
            password = self.password
            print("Attempting to establist SHH connection")
            self.shh.connect(addr, username=username,  password=password)
            print("SHH Connection established")
        except:
            print("Unable to establish a connection")

    def ask_hostname(self):
        """Ask the RPI Server to start running a server file"""
        stdin, stdout, stderror = self.shh.exec_command("hostname -I")
        print(stdout.readlines())

    def start_server(self):
        #stdin, stdout, stderror = self.shh.exec_command("cd ~/Documents/PostureTracker_Project")
        stdin, stdout, stderror = self.shh.exec_command(
            "python3 /PSAT_Server/posturalCam_master_NETWORK.py")

    def shutdown_server_pi(self):
        stdin, stdout, stderror = self.shh.exec_command("sudo shutdown now")
        print("stdout: {}\nstderror: {}".format(
            stdout.readlines(), stderror.readlines()))

    def restart_server_pi(self):
        stdin, stdout, stderror = self.shh.exec_command("sudo reboot now")
        print("stdout: {}\nstderror: {}".format(
            stdout.readlines(), stderror.readlines()))

    def send_command(self, cmd):
        stdin, stdout, stderror = self.shh.exec_command(cmd)
        print("stdout: {}\nstderror: {}".format(
            stdout.readlines(), stderror.readlines()))


if __name__ == '__main__':
    my_shh = shh_client()
    my_shh.ask_hostname()
    my_shh.restart_server_pi()
    # my_shh.start_server()
    # my_shh.shutdown_server_pi()
