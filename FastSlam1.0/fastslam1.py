from utils.read_utils import read_world, read_sensor_data
from utils.fastSLAM_utils import prediction_step, correction_step, resample

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
	# read ground truth world (landmarks)
	landmarks=read_world('./data/world.dat')
	gt_lx=np.array([[landmark['x'],landmark['y']] for landmark in landmarks])

	# plot ground truth landmarks for reference
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(gt_lx[:,0], gt_lx[:,1], s=[100]*gt_lx.shape[0], marker='+')
	ax.set_xlim([-1,12])
	ax.set_ylim([-1,12])
	plt.show(block=False)

	# read odometry and sensor data
	data=read_sensor_data('./data/sensor_data.dat')

	num_particles=100
	# initialize particle set
	particle_set=[]
	for i in range(num_particles):
		particle={'pose':{}, 'map_estimate':{}, 'weight':1.0/num_particles}
		particle['pose']['x'], particle['pose']['y'], particle['pose']['th']=0.0,0.0,0.0
		particle_set.append(particle)

	# simulate motion
	odometry_noise=[0.005, 0.01, 0.005]
	p_scatter=ax.scatter([particle['pose']['x'] for particle in particle_set], [particle['pose']['y'] for particle in particle_set], s=[10]*num_particles, color='#00ff00')
	bp = ax.scatter([particle_set[0]['pose']['x']], [particle_set[0]['pose']['y']], s=[10], color='#ff0000')
	plt.pause(0.1)
	i=0
	for data_t in data:
		# step 1: prediction
		odometry=data_t['odometry']
		particle_set=prediction_step(particle_set, odometry, noise=odometry_noise)

		# step 2: correction
		sensor_readings=data_t['sensor']
		correction_step(particle_set, sensor_readings)

		# get best particle for displaying its corresponding state
		weights = [particle['weight'] for particle in particle_set]
		best_particle_id = np.argmax(weights)

		# step 3: resampling
		particle_set=resample(particle_set)

		p_scatter.set_offsets(np.array([[particle['pose']['x'], particle['pose']['y']] for particle in particle_set]))
		bp.set_offsets(np.array([particle_set[best_particle_id]['pose']['x'], particle_set[best_particle_id]['pose']['y']]))
		est_lx=np.vstack([l.mu.T for l in particle_set[best_particle_id]['map_estimate'].values()])
		est_lx_scatter=ax.scatter(est_lx[:,0], est_lx[:,1], marker='*', s=[10]*est_lx.shape[0], color='#ff0000')
		observation_lines=[]
		for reading in sensor_readings:
			landmark_id = reading['id']
			pline = ax.plot([particle_set[best_particle_id]['pose']['x'], particle_set[best_particle_id]['map_estimate'][landmark_id].mu[0,0]], [particle_set[best_particle_id]['pose']['y'], particle_set[best_particle_id]['map_estimate'][landmark_id].mu[1,0]] , 'k-', linewidth=0.2)
			observation_lines.append(pline)
		plt.pause(0.1)
		plt.savefig('./plots/plot_{0:03d}.png'.format(i))
		i=i+1
		est_lx_scatter.remove()
		for pline in observation_lines:
			pline.pop().remove()