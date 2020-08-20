from utils.read_robotlaser import read_robotlaser, rangedata_to_xy
from utils.grid_utils import x2i, xy2ij, bresenham_raytrace
from matplotlib import pyplot as plt
import numpy as np

def prob2logOdds(p):
	return np.log(p/(1.0-p))

def logOdds2prob(l):
	el=np.exp(-l)
	return 1.0/(1.0+el)

if __name__=="__main__":
	data="./data/csail.log"
	data=read_robotlaser(data)

	p0=0.5
	l0=prob2logOdds(p0)
	pFree=0.35
	lFree=prob2logOdds(pFree)
	pOcc=0.9
	lOcc=prob2logOdds(pOcc)
	# gridmap
	mapRes=0.1
	border=30
	xMin,xMax,yMin,yMax=np.finfo(float).max,np.finfo(float).min,np.finfo(float).max,np.finfo(float).min
	for i in range(len(data)):
		robot_pose=data[i]["robot_pose"]
		xMin=min(xMin,robot_pose[0])
		xMax=max(xMax,robot_pose[0])
		yMin=min(yMin,robot_pose[1])
		yMax=max(yMax,robot_pose[1])
	mapBox=[xMin-border,yMin-border,xMax+border,yMax+border]
	mapSize=[x2i(mapBox[3],mapBox[1],mapRes)+1, x2i(mapBox[2],mapBox[0],mapRes)+1]

	gridmap=l0*np.ones(mapSize)
	for i in range(0,len(data)):
		update=np.zeros_like(gridmap)
		
		robot_pose=data[i]["robot_pose"]
		#convert robot pose in world coords (x,y in m) to map coords (col, row indices)
		c0,r0=xy2ij(robot_pose[0],robot_pose[1],mapBox[0],mapBox[1],mapRes)

		points_world,valid_ids=rangedata_to_xy(data[i],max_range=30)
		points_world=points_world[:,valid_ids]

		# 	1. each valid point in points_world is a beam end point
		# 	2. get free cells along the ray, convert them to map indices and update those map log odds using free probability pFree
		#		pFree = probability that a free cell is occupied
		#	3. for the beam end point, convert it to map index and update map log odds using occupied probability pOcc
		#		poOcc = probability that an occupied cell is occupied
		for j in range(points_world.shape[1]):
			c,r=xy2ij(points_world[0,j],points_world[1,j],mapBox[0],mapBox[1],mapRes)
			X=bresenham_raytrace(c0,r0,c,r)
			for k in range(0,X.shape[1]):
				if X[0,k]==c and X[1,k]==r:
					continue
				else:
					update[X[1,k],X[0,k]]=lFree
			update[r,c]=lOcc
		update -= l0*np.ones_like(gridmap)
		gridmap += update

	plt.imshow(np.ones_like(gridmap)-logOdds2prob(gridmap),cmap="gray")
	plt.show()