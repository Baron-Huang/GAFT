##标注ADS-CAM时候，选最优CAM

import os
from skimage import io

if __name__ == '__main__':
    img_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_greece\IDANet\Images\I'
    cam_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_greece\IDANet\CAMs\I'
    img_dir = os.listdir(img_path)
    cam_dir = os.listdir(cam_path)
    new_cam_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_greece\IDANet\New_CAMs\I'

    count = 0
    for i in img_dir:
        for j in cam_dir:
            if i == j:
               cam_inter = io.imread(cam_path + r'\\' + j)
               io.imsave(new_cam_path + r'\\' + j, cam_inter)
               count += 1
               print(count)
