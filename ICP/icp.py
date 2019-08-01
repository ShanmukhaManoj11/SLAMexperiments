from icp_3d_utils import *

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
	x_hist=np.zeros((x.shape[0],max_iterations+1))
	x_hist[:,0]=np.reshape(x,(x.shape[0],))
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
		x_hist[:,it+1]=np.reshape(x,(x.shape[0],))
	return x,errors,x_hist