import os
import numpy as np

basedir = "D:\NeRF\datasets\llff"

poses_arr = np.load(os.path.join(basedir, 'poses.npy'))
print(poses_arr)