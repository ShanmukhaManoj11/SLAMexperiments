import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from icp import *
from scipy.optimize import minimize
import time

#generate sample points
n_points=100
sample_std=5
sample_mean=3
P_world=sample_std*np.random.randn(3,n_points)+sample_mean

#assign true world to robot transformation
x_true=np.array([[10],[5],[-6],[np.pi/3],[np.pi/4],[-np.pi/6]],dtype=np.float32)
T_WtoR=v2T(x_true)

#transform sample points to robot coordinate frame
P_robot_true=transformPoints(T_WtoR,P_world)
#add noise and let these be the measurements in robot coordinates
Z=P_robot_true+np.random.randn(3,n_points)*0.1

#initial guess
x_init=np.array([[0],[0],[0],[0],[0],[0]],dtype=np.float32)
max_iterations=10
[x_est,errors,x_hist]=gaussNewtonOptimizer(errorAndJacobian,x_init,P_world,Z,max_iterations)

T_WtoR_est=v2T(x_est)
P_robot_est=transformPoints(T_WtoR_est,P_world)

#animation
def update(i,x_hist,scat):
	T_WtoR_i=v2T( np.reshape( x_hist[:,i], (x_hist[:,i].shape[0],1) ) )
	P_robot_i=transformPoints(T_WtoR_i,P_world)
	scat._offsets3d=(P_robot_i[0,:],P_robot_i[1,:],P_robot_i[2,:])

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(P_robot_true[0,:],P_robot_true[1,:],P_robot_true[2,:],c='b')

T_WtoR_0=v2T( np.reshape( x_hist[:,0], (x_hist[:,0].shape[0],1) ) )
P_robot_0=transformPoints(T_WtoR_0,P_world)
scat=ax.scatter(P_robot_0[0,:],P_robot_0[1,:],P_robot_0[2,:],c='r')
ani=animation.FuncAnimation(fig,update,frames=xrange(1,max_iterations+1),
	fargs=(x_hist,scat),repeat=False)
plt.show()

print('true transform - est transform')
print(np.hstack([x_true,x_est]))
