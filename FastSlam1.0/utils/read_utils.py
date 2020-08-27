def read_world(world_file):
	#########################################################################################
	# @ world is represented as list of landmarks 											#
	# each line in the file represents location of a landmark in the format					#
	# 	id x y 																				#
	#########################################################################################
	landmarks=[]
	with open(world_file,'r') as f:
		lines=f.readlines()
		for line in lines:
			data=line.split()
			landmark={'id':int(data[0]), 'x':float(data[1]), 'y':float(data[2])}
			landmarks.append(landmark)
	return landmarks

def read_sensor_data(sensor_data_file):
	#########################################################################################
	# @ each line in sensor data file is either ODOMETRY or SENSOR reading					#
	# 	ODOMETRY r1 t r2 																	#
	#	SENSOR id range bearing 															#
	#########################################################################################
	data=[]
	with open(sensor_data_file, 'r') as f:
		lines=f.readlines()
		for line in lines:
			v=line.split()
			if v[0]=='ODOMETRY':
				data_t={}
				data_t['odometry']={}
				data_t['odometry']['r1']=float(v[1])
				data_t['odometry']['t']=float(v[2])
				data_t['odometry']['r2']=float(v[3])
				data.append(data_t)
			elif v[0]=='SENSOR':
				if 'sensor' not in data[-1]:
					data[-1]['sensor']=[]
				data_t_sensor={}
				data_t_sensor['id']=int(v[1])
				data_t_sensor['range']=float(v[2])
				data_t_sensor['bearing']=float(v[3])
				data[-1]['sensor'].append(data_t_sensor)
		return data