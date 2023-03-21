import numpy as np
import cv2 as cv
import glob
from configparser import ConfigParser
import os


class CameraCalibration():
    def __init__(self,img_path,config_path):
        self.left_img_path = os.path.join(img_path,"left_im/*.jpg")
        self.right_img_path = os.path.join(img_path,"right_im/*.jpg")
        self.config_path = config_path
        self.left_mtx, self.left_dist, = self.CalibrateCamera(self.left_img_path)
        self.right_mtx, self.right_dist = self.CalibrateCamera(self.right_img_path)

    def CalibrateCamera(self, path):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob(path)
        print(path)
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (7,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                # cv.drawChessboardCorners(img, (7,6), corners2, ret)
                # cv.imshow('img', img)
                # cv.waitKey(500)
        # cv.destroyAllWindows()
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist
    def UpdateConfig(self):
        left_fx = str(self.left_mtx[0][0])
        left_fy = str(self.left_mtx[1][1])
        left_cx = str(self.left_mtx[0][2])
        left_cy = str(self.left_mtx[1][2])

        left_k1 = str(self.left_dist[0][0])
        left_k2 = str(self.left_dist[0][1])
        left_k3 = str(self.left_dist[0][4])
        left_p1 = str(self.left_dist[0][2])
        left_p2 = str(self.left_dist[0][3])

        right_fx = str(self.right_mtx[0][0])
        right_fy = str(self.right_mtx[1][1])
        right_cx = str(self.right_mtx[0][2])
        right_cy = str(self.right_mtx[1][2])

        right_k1 = str(self.right_dist[0][0])
        right_k2 = str(self.right_dist[0][1])
        right_k3 = str(self.right_dist[0][4])
        right_p1 = str(self.right_dist[0][2])
        right_p2 = str(self.right_dist[0][3])


        #Read config.ini file
        config_object = ConfigParser()
        config_object.read(self.config_path)

        #Get the USERINFO section
        # left_cam_2k = config_object["LEFT_CAM_2K"]
        # right_cam_2k = config_object["RIGHT_CAM_2K"]
        # left_cam_fhd = config_object["LEFT_CAM_FHD"]
        # right_cam_fhd = config_object["RIGHT_CAM_FHD"]
        left_cam_hd = config_object["LEFT_CAM_HD"]
        right_cam_hd = config_object["RIGHT_CAM_HD"]
        # left_cam_vga = config_object["LEFT_CAM_VGA"]
        # right_cam_vga = config_object["RIGHT_CAM_VGA"]

        #Update the data
        left_cam_hd["fx"] = left_fx
        left_cam_hd["fy"] = left_fy
        left_cam_hd["cx"] = left_cx
        left_cam_hd["cy"] = left_cy
        left_cam_hd["k1"] = left_k1
        left_cam_hd["k2"] = left_k2
        left_cam_hd["k3"] = left_k3
        left_cam_hd["p1"] = left_p1
        left_cam_hd["p2"] = left_p2

        right_cam_hd["fx"] = right_fx
        right_cam_hd["fy"] = right_fy
        right_cam_hd["cx"] = right_cx
        right_cam_hd["cy"] = right_cy
        right_cam_hd["k1"] = right_k1
        right_cam_hd["k2"] = right_k2
        right_cam_hd["k3"] = right_k3
        right_cam_hd["p1"] = right_p1
        right_cam_hd["p2"] = right_p2

        #Write changes back to file
        with open(self.config_path, 'w') as conf:
            config_object.write(conf)
