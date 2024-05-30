import sys
sys.path.insert(1,'./1_rgh_class')
import rgh_class as rgh
import numpy as np

x=np.linspace(0,5,1000)
z=np.linspace(0,5,1000)
Gridx,Gridz=np.meshgrid(x,z)
y=0.5*np.sin(Gridx)+0.5*np.sin(Gridz)
try:
  surface=rgh.rgh(x,z,y)
except:
  assert False , "rgh instance is not properly created"
DF_surface_statistics=surface.print_stat()
assert ~DF_surface_statistics.isnull().values.any()