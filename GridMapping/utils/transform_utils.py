import numpy as np

def v2t(v):
	#########################################################################
	# @ compute corresponding relative to world coordinates transformation 	#
	# 	for a given pose vector (x,y,theta) 								#
	#########################################################################
	c,s=np.cos(v[2]),np.sin(v[2])
	T=np.array([[c,-s,v[0]],[s,c,v[1]],[0,0,1]])
	return T

def t2v(T):
	#########################################################################
	# @ compute corresponding pose vector from 							 	#
	# 	relative to world coordinates transformation matrix of form			#
	# 	[cos(th) -sin(th) x] 												#
	# 	[sin(th)  cos(th) y] 												#
	# 	[0        0       1] 												#
	#########################################################################
	v=np.zeros((3,1))
	v[0:2,1]=T[0:2,2]
	v[2,1]=np.arctan2(T[1,0],T[0,0])
	return v