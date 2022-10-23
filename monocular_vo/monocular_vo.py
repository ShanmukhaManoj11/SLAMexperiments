import os
import cv2
from kitti_utils import *
from vo_module import VOModuleMonocular

BASE_DIR = "../../data_odometry_gray/dataset/sequences"
SEQUENCE = "00"
VIS = True

def main():
    vo = VOModuleMonocular(vis=VIS)
    imageFiles = sorted(os.listdir(os.path.join(BASE_DIR, SEQUENCE, "image_0")))
    for imgFile in imageFiles[:500]:
        img = cv2.imread(os.path.join(BASE_DIR, SEQUENCE, "image_0", f"{imgFile}"), 0)
        vo.processFrame(img)
    if VIS:
        cv2.destroyAllWindows()

    gtPoses = KittiUtils.parseGTPoses(os.path.join(BASE_DIR, SEQUENCE, f"{SEQUENCE}.txt"))
    KittiUtils.plotPath([gtPoses[:500], np.array(vo.poses)])
    # KittiUtils.plotPath([gtPoses])

if __name__ == "__main__":
    main()