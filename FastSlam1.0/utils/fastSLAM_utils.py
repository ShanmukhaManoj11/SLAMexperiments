import numpy as np

def normalize_angle(th):
	#########################################################################################
	# @ normalize input angle between -pi and pi 											#
	#########################################################################################
	while th>np.pi:
		th = th - 2.0*np.pi
	while th<=-np.pi:
		th = th + 2.0*np.pi
	return th

def odometry_motion_model(pose, odometry, noise=[0.005, 0.01, 0.005]):
	#########################################################################################
	# @ motion model to estimate/ predict next state given odometry information 			#
	# @ p(x_<t+1> | x_<t>, u_<t>) 															#
	# input: 																				#
	#	1. pose = dict with keys 'x', 'y', 'th'												#
	#	2. odometry = dict with keys 'r1', 't', 'r2' 	 									#
	#	3. noise = noise standard deviations for r1, t and r2 respectively 					#
	# output: 																				#
	#	1. pose_ = predicted pose given odometry 											#
	#########################################################################################
	r1 = np.random.normal(odometry['r1'], noise[0])
	t = np.random.normal(odometry['t'], noise[1])
	r2 = np.random.normal(odometry['r2'], noise[2])
	pose_ = {}
	pose_['x'] = pose['x'] + t*np.cos(pose['th']+r1)
	pose_['y'] = pose['y'] + t*np.sin(pose['th']+r1)
	pose_['th'] = normalize_angle(pose['th'] + r1 + r2)
	return pose_

def prediction_step(particle_set, odometry, noise=[0.005, 0.01, 0.005]):
	#########################################################################################
	# @ apply motion model to each particle in the particle set 							#
	# particle_set = list of particles, 													#
	# 	each particle is a dict with keys 'pose', 'map_estimate', 'weight'					#
	#		'pose' -> dict with keys: 'x', 'y', 'th' 										#
	#		'map_estimate' -> dict of EKF estimators for the landmarks with ids as keys		#
	#		'weight' -> associated weight for the particle 									#
	#########################################################################################
	particle_set_=[]
	num_particles=len(particle_set)
	for particle in particle_set:
		pose_ = odometry_motion_model(particle['pose'], odometry, noise=noise)
		# assuming landmarks are static
		map_estimate_ = particle['map_estimate']
		w_ = particle['weight']
		particle_={'pose':pose_, 'map_estimate': map_estimate_, 'weight':w_}
		particle_set_.append(particle_)
	return particle_set_

class EKF_StateEstimator(object):
	def __init__(self, mu, H=np.eye(2)):
		self.I = np.eye(2)
		self.Qt = np.eye(2)*0.1
		self.mu = mu
		Hinv = np.linalg.pinv(H)
		self.Sigma = np.matmul(Hinv, np.matmul(self.Qt, Hinv.T))

	def correction_step(self, z_pred, H, z):
		Q = np.matmul(H, np.matmul(self.Sigma, H.T)) + self.Qt
		Qinv = np.linalg.pinv(Q)
		K = np.matmul(self.Sigma, np.matmul(H.T, Qinv))
		zdelta = z-z_pred
		zdelta[1,0] = normalize_angle(zdelta[1,0])
		self.mu = self.mu + np.matmul(K, zdelta)
		self.Sigma = np.matmul(self.I - np.matmul(K, H), self.Sigma)
		w = 1/np.sqrt(2.0*np.pi*np.linalg.det(Q)) * np.exp(-0.5*np.matmul(zdelta.T, np.matmul(Qinv, zdelta)))
		return w

def measurement_model(particle, sensor_reading):
	#########################################################################################
	# @ given pose and observed landmark ids (data association) return 						#
	# @ predcited measurements and jacobian H 												#	
	# @ essentially computes p(z_<t> | x_<t>, m)	 										#
	# particle = dict containing robot 'pose' and 'map_estimate'							#
	# sensor_reading = dict with keys 'id', 'range', 'bearing' 								#
	#########################################################################################
	pose = particle['pose']
	landmark_id = sensor_reading['id']
	if landmark_id not in particle['map_estimate']:
		z_pred = np.array([[sensor_reading['range']], [sensor_reading['bearing']]])
		H = np.zeros((2,2))
		H[0,0] = np.cos(pose['th']+sensor_reading['bearing'])
		H[0,1] = np.sin(pose['th']+sensor_reading['bearing'])
		H[1,0] = -np.sin(pose['th']+sensor_reading['bearing'])/sensor_reading['range']
		H[1,1] = np.cos(pose['th']+sensor_reading['bearing'])/sensor_reading['range']
		return z_pred, H
	landmark_pose = particle['map_estimate'][landmark_id].mu
	range_pred = np.sqrt((landmark_pose[0,0] - pose['x'])**2 + (landmark_pose[1,0] - pose['y'])**2)
	bearing_pred = normalize_angle(np.arctan2(landmark_pose[1,0]-pose['y'], landmark_pose[0,0]-pose['x']) - pose['th'])
	z_pred = np.array([[range_pred],[bearing_pred]])

	H = np.zeros((2,2))
	H[0,0] = (landmark_pose[0,0]-pose['x'])/range_pred
	H[0,1] = (landmark_pose[1,0]-pose['y'])/range_pred
	H[1,0] = (pose['y']-landmark_pose[1,0])/range_pred**2
	H[1,1] = (landmark_pose[0,0]-pose['x'])/range_pred**2

	return z_pred, H

def correction_step(particle_set, sensor_readings):
	num_particles=len(particle_set)
	for particle in particle_set:
		for sensor_reading in sensor_readings:
			landmark_id = sensor_reading['id']
			z_pred, H = measurement_model(particle, sensor_reading)
			if landmark_id not in particle['map_estimate']:
				lx = particle['pose']['x'] + sensor_reading['range']*np.cos(particle['pose']['th']+sensor_reading['bearing'])
				ly = particle['pose']['y'] + sensor_reading['range']*np.sin(particle['pose']['th']+sensor_reading['bearing'])
				particle['map_estimate'][landmark_id]=EKF_StateEstimator(np.array([[lx],[ly]]), H=H)
				particle['weight']=1.0/num_particles
			else:
				w=particle['map_estimate'][landmark_id].correction_step(z_pred, H, np.array([[sensor_reading['range']],[sensor_reading['bearing']]]))
				particle['weight'] = particle['weight']*w.squeeze()

def resample(particle_set):
	weights=np.array([particle['weight'] for particle in particle_set])
	weights=weights/np.sum(weights)

	num_particles=len(particle_set)
	r=np.random.rand()/num_particles
	i=0
	c=weights[i]
	particle_set_=[]
	for m in range(num_particles):
		U = r+m/num_particles
		while U>c:
			i=i+1
			c=c+weights[i]
		particle_={'pose':particle_set[i]['pose'], 'map_estimate':particle_set[i]['map_estimate'], 'weight':1.0/num_particles}
		particle_set_.append(particle_)
	return particle_set_