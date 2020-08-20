import numpy as np
from .transform_utils import v2t

def read_robotlaser(file_path):
	#########################################################################################################
	# @ reads laser data from files 																		#
	# data is expected to be of the from 																	#
	# ROBOTLASER1 0 start-angle fov angle-res max-range accuracy remission-mode num-readings 				#
	# 	[space separated readings] num-remissions [space separated remission values] laser-pose-x 			#
	# 	laser-pose-y laser-pose-theta robot-pose-x robot-pose-y robot-pose-theta tv rv forward side turn 	#
	#	timestamp b21 float-value 																			#
	#########################################################################################################
	data=[]
	with open(file_path,'r') as f:
		lines=f.readlines()
		for line in lines:
			currentReading={}
			values=line.split()
			if values[0] != "ROBOTLASER1":
				continue
			currentReading["start_angle"]=float(values[2])
			currentReading["fov"]=float(values[3])
			currentReading["angle_res"]=float(values[4])
			currentReading["max_range"]=float(values[5])
			num_readings=int(values[8])
			currentReading["ranges"]=[]
			for k in range(num_readings):
				currentReading["ranges"].append(float(values[9+k]))
			i=9+num_readings
			num_remissions=int(values[i])
			i+=1+num_remissions
			currentReading["laser_pose"]=list(map(float,values[i:i+3]))
			i+=3
			currentReading["robot_pose"]=list(map(float,values[i:i+3]))
			i+=3
			i+=5 # skip tv, rv, forward, side, turn
			currentReading["timestamp"]=float(values[i])
			data.append(currentReading)
		return data

def rangedata_to_xy(reading,max_range=15):
	#########################################################################################################
	# @ converts range data in the reading to x,y points 													#
	# Sensor provides returns for linspace of angles starting and start_angle and spaced at angle_res 		#
	# return points in world frame and valid_inds of points that have range values in range [0,max_range] 	#
	#########################################################################################################
	start_angle=reading["start_angle"]
	angle_res=reading["angle_res"]
	ranges=np.array(reading["ranges"])
	num_ranges=len(ranges)
	end_angle=start_angle+num_ranges*angle_res
	angles=np.arange(start_angle,end_angle,angle_res)

	max_range=min(max_range,reading["max_range"])
	valid_ids=(ranges>0) & (ranges<max_range)

	points_laser=np.vstack([ranges*np.cos(angles), ranges*np.sin(angles), np.ones_like(ranges)])
	T_laser2world=v2t(reading["laser_pose"])
	points_world=np.matmul(T_laser2world,points_laser)
	return points_world, valid_ids