# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:46:28 2023

@author: Jiasheng
"""

#from rgh_class import *
import rgh_class as rgh
import scipy as sp
from scipy.io import loadmat
import numpy as np
F=loadmat("Surface.mat")
x=F["x"].reshape(-1,1)
y=F["y"]
z=F["z"].reshape(-1,1)
roughness=rgh.rgh(x,z,y)