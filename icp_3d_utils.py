import numpy as np

def R(th,axis=0):
	c=np.cos(th)
	s=np.sin(th)
	if axis==0:
		R=np.array([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]],dtype=np.float32);
	elif axis==1:
		R=np.array([[c,0.0,s],[0.0,1.0,0.0],[-s,0.0,c]],dtype=np.float32);
	elif axis==2:
		R=np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]],dtype=np.float32);
	else:
		raise Exception('axis should be 0, 1 or 2')
	return R

def dR(th,axis=0):
	dc=-np.sin(th)
	ds=np.cos(th)
	if axis==0:
		R=np.array([[0.0,0.0,0.0],[0.0,dc,-ds],[0.0,ds,dc]],dtype=np.float32);
	elif axis==1:
		R=np.array([[dc,0.0,ds],[0.0,0.0,0.0],[-ds,0.0,dc]],dtype=np.float32);
	elif axis==2:
		R=np.array([[dc,-ds,0.0],[ds,dc,0.0],[0.0,0.0,0.0]],dtype=np.float32);
	else:
		raise Exception('axis should be 0, 1 or 2')
	return R

def v2T(v):
	T=np.eye(4)
	T[0:3,0:3]=np.matmul(np.matmul(R(v[3,0],axis=0),R(v[4,0],axis=1)),R(v[5,0],axis=2))
	T[0:3,3]=v[0:3,0]
	return T;

def transformPoints(T,p):
	'''
	p.shape=[3,n] - n 3-dimensional points to be transformed by transformation T
	T.shape=[4,4] = [R;0|t;1] - homogeneous transformation mat
	'''
	pH=np.ones( (p.shape[0]+1,p.shape[1]) )
	pH[0:p.shape[0],:]=p #points p in homogeneous coordinates
	qH=np.matmul(T,pH) #transformed points q in homogenoues coordinates
	return qH[0:p.shape[0],:]