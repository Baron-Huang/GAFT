#通过JSON文件生产真实mask时候，从子文件中选取mask到根目录下

import os
from skimage import io

if __name__ == '__main__':
    view_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_china\Mask_View\III'
    view_dir = os.listdir(view_path)
    mask_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_china\Masks\III'

    count = 0
    for i in view_dir:
        view_inter = io.imread(view_path + r'\\' + i + r'\\' + 'label.png')
        io.imsave(mask_path + r'\\' + i + '.png', view_inter)
        count += 1
        print(count)

    pass


