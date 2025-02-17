import numpy as np
import cv2

class SIFTKeypointsDiscriptor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
           
    def generate_keypoints_descriptors(self, img:np.ndarray):
        """input should be a single channel numpy array output is the keypoints and their descriptors"""
        self.img = img
        _keypoints, _descriptors = self.sift.detectAndCompute(img, None)
        return _keypoints, _descriptors
    
    @staticmethod
    def getnp(_cv2Object:list):
    
    @staticmethod
    def bgr2gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def draw_keypoints(self, save=False):
        out=cv2.drawKeypoints(self.img, self.keypoints, self.img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        while True:
            cv2.imshow("SIFT Keypoints", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        if save:
            cv2.imwrite("sift.png", out)

img = cv2.imread("test.jpg")

# apply sift
sift = SIFTKeypointsDiscriptor()
img = sift.bgr2gray(img)
sift.generate_keypoints_descriptors(img)

# draw keypoints
sift.draw_keypoints() 
