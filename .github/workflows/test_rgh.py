import sys
sys.path.insert(1,'./1_rgh_class')
import rgh_class as rgh
import numpy as np

x=np.linspace(0,4,1000)
z=np.linspace(0,4,1000)
Gridx,Gridz=np.meshgrid(x,z)
y=0.5*np.sin(Gridx*np.pi)+0.5*np.sin(Gridz*np.pi)
try:
  surface=rgh.rgh(x,z,y)
except:
  assert False , "rgh instance is not properly created"
metrics=[surface.Lx,surface.Lz,surface.sk,surface.ku,surface.krms,surface.kmd,surface.kt,surface.k99,surface.ra,surface.por,surface.ESx,surface.ESz,surface.incx,surface.incz]
assert ~any(np.isnan(metrics)) , "went wrong when calculating roughness metrics"