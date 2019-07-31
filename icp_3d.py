import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from icp_3d_utils import *
from scipy.optimize import minimize

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
Z=P_robot_true+np.random.randn(3,n_points)

def errorAndJacobian(x,p,z):
	# current state x, point p in world coordinates, corresponding measurement in robot coordinates
	T=v2T(x);
	z_est=transformPoints(T,p)
	error=z_est-z
	J=np.zeros((3,6))
	J[0:3,0:3]=np.eye(3);
	Rx,dRx=R(x[3],axis=0),dR(x[3],axis=0)
	Ry,dRy=R(x[4],axis=1),dR(x[4],axis=1)
	Rz,dRz=R(x[5],axis=2),dR(x[5],axis=2)
	J[0:3,3:4]=np.matmul(np.matmul(np.matmul(dRx,Ry),Rz),p)
	J[0:3,4:5]=np.matmul(np.matmul(np.matmul(Rx,dRy),Rz),p)
	J[0:3,5:6]=np.matmul(np.matmul(np.matmul(Rx,Ry),dRz),p)
	return error,J

def gaussNewtonOptimizer(f,x_init,P,Z,max_iterations):
	x=x_init
	errors=np.zeros((1,max_iterations))
	for it in range(max_iterations):
		H=np.zeros((6,6))
		b=np.zeros((6,1))
		error_=0.0
		for i in range(P.shape[1]):
			[error,J]=f(x,np.reshape(P[:,i],(P.shape[0],1)),np.reshape(Z[:,i],(Z.shape[0],1)))
			error_=error_+np.matmul(np.transpose(error),error)
			H=H+np.matmul(np.transpose(J),J)
			b=b+np.matmul(np.transpose(J),error)
		errors[0,it]=error_
		dx=np.linalg.solve(-H,b)
		x=x+dx;
	return x,errors

x_init=np.array([[0],[0],[0],[0],[0],[0]],dtype=np.float32)
max_iterations=10
[x_est,errors]=gaussNewtonOptimizer(errorAndJacobian,x_init,P_world,Z,max_iterations)

T_WtoR_est=v2T(x_est)
P_robot_est=transformPoints(T_WtoR_est,P_world)

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(P_robot_true[0,:],P_robot_true[1,:],P_robot_true[2,:],c='b')
ax.scatter(P_robot_est[0,:],P_robot_est[1,:],P_robot_est[2,:],c='r')
plt.show()

plt.plot(np.arange(1,max_iterations),errors[0,1:])
plt.show()

print(np.hstack([x_true,x_est]))
