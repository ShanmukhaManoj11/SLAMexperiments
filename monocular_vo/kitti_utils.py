import numpy as np
import open3d

class KittiUtils(object):
    """KITTI odometry dataset utils
    """
    P0 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
                   [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
                   [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]]) # projection matrix for camera 0
    K0 = P0[:, :3] # instrinsics for camera 0
    K0inv = np.linalg.inv(K0)

    P1 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02],
                   [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
                   [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]]) # projection matrix for camera 1

    f = P0[0,0] # focal length
    B = -P1[0,3]/f # Baseline distance (m) between stereo pair camera 0 (left) and camera 1 (right)

    def __init__(self):
        self.super().__init__

    @staticmethod
    def parseGTPoses(filepath):
        with open(filepath, "r") as f:
            data = f.readlines()
        gtPoses = []
        for pose in data:
            pose = list(map(float, pose.strip().split()))
            pose = np.array(pose).reshape((3, 4))
            gtPoses.append(pose)
        return np.array(gtPoses) # [N, 3, 4]

    @staticmethod
    def plotPath(paths):
        geometries = []
        for poses in paths:
            color = [np.random.random(), np.random.random(), np.random.random()]
            points = poses[:, :, 3]
            colors = np.array([color]*points.shape[0])
            colors[0, :] = [0, 1, 0] # start point
            colors[-1, :] = [1, 0, 0] # end point
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.colors = open3d.utility.Vector3dVector(colors)

            lines = np.array([[i-1, i] for i in range(1, points.shape[0])])
            lineSet = open3d.geometry.LineSet()
            lineSet.points = open3d.utility.Vector3dVector(points)
            lineSet.lines = open3d.utility.Vector2iVector(lines)

            geometries.append(pcd)
            geometries.append(lineSet)
            
        open3d.visualization.draw_geometries(geometries)