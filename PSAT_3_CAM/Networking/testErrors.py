# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:33:40 2016

@author: pi
"""


import sys

try:
    
    int("d")

except :
    
    print("Unexpected Error: ", sys.exc_info())