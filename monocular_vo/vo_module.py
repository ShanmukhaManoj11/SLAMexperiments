import cv2
import numpy as np
from kitti_utils import KittiUtils

FLANN_INDEX_LSH = 6

class VOModuleMonocular(object):
    """ ORB feature based visual odometry computation (monocular).
    """

    def __init__(self, vis = False):
        super().__init__()
        self.orb = cv2.ORB_create(3000) 
        # BF Matcher
        self.ftMatcher = cv2.BFMatcher() 
        # FLANN based matcher
        # indexParams= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
        # searchParams = dict(checks=50)
        # self.ftMatcher = cv2.FlannBasedMatcher(indexParams, searchParams)
        self.frames = []
        self.keypoints = []
        self.descriptors = []
        self.vis = vis
        self.poses = []
    
    def processFrame(self, frame):
        """ Process new frame
        1. compute features/ keypoints in the input frame,
        2. match features with previous frame
        3. compute essential matrix with matched keypoints
           ref: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga0b166d41926a7793ab1c351dbaa9ffd4
        4. decompose essential matrix into translation and rotation terms
           ref: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d

        @param frame input image to process
        """
        self.frames.append(frame)

        # 1. compute features/ keypoints in the input frame
        kp, des = self.orb.detectAndCompute(frame, None)
        self.keypoints.append(kp)
        self.descriptors.append(des)

        if len(self.frames) == 1: # processing first frame
            self.poses.append(np.eye(3, 4))
            return

        # 2. match features with previous frame
        kpPrev, desPrev = self.keypoints[-2], self.descriptors[-2]
        matches = self.ftMatcher.knnMatch(desPrev, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        matches = good
        
        if self.vis:
            dispParams = dict(matchColor = -1, singlePointColor = None,
                matchesMask = None, flags = 2)
            disp = cv2.drawMatches(self.frames[-2], self.keypoints[-2], self.frames[-1], self.keypoints[-1], 
                matches, None, **dispParams)
            disp = cv2.resize(disp, self.frames[-1].shape[:2][::-1])
            cv2.imshow("matches", disp)
            cv2.waitKey(5)
        
        # 3. compute essential matrix from matched points
        p1 = np.float64([kpPrev[m.queryIdx].pt for m in matches])
        p2 = np.float64([kp[m.trainIdx].pt for m in matches])
        # ref: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga0b166d41926a7793ab1c351dbaa9ffd4
        E, _ = cv2.findEssentialMat(p1, p2, KittiUtils.K0, threshold=1.0)
        
        # 4. decompose E into R, t
        _, R, t, _ = cv2.recoverPose(E, p1, p2, KittiUtils.K0)
        T = self.getTransformationMat(R, t)
        Tw = np.concatenate([self.poses[-1], np.ones((1, 4))], axis=0) @ np.linalg.inv(T)
        self.poses.append(Tw[:3, :])

    def getTransformationMat(self, R: np.array, t:np.array) -> np.array:
        """ Get transformation matrix from rotation (R) and translation (t) parts

        @param R rotation portion, np array of shape [3, 3]
        @param t translation portion, np array of shape [3, 1]

        @return transformation matrix, np array of shape [4, 4]
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3:] = t
        return T
