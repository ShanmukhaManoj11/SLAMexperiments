import numpy as np

def x2i(x,x0,map_res):
	#####################################################################
	# @ given start corner x0/y0 of the grid map and map resolution, 	#
	# 	convert the given world x/y value to the map index 				#
	#####################################################################
	return np.int(np.floor((x-x0)/map_res))

def xy2ij(x,y,x0,y0,map_res):
	#################################################################
	# @ convert (x,y) in world coordinates to map indices (i,j) 	#
	#################################################################
	return np.int(x2i(x,x0,map_res)), np.int(x2i(y,y0,map_res))

def bresenham_raytrace(x0,y0,x1,y1):
	steep = (abs(y1-y0) > abs(x1-x0))
	if steep:
		x0,x1,y0,y1=y0,y1,x0,x1

	if x0>x1:
		x0,y0,x1,y1=x1,y1,x0,y0

	dx=x1-x0
	dy=abs(y1-y0)
	X=np.zeros((2,dx+1),dtype=int)
	ystep = 1 if y0<y1 else -1
	err=0
	x,y=x0,y0
	for n in range(0,dx+1):
		X[:,n]=[x,y]
		x=x+1
		err=err+dy
		if 2*err>=dx:
			y=y+ystep
			err=err-dx
	if steep:
		X=X[::-1,:]
	return X