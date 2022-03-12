import math
import os
import time
from pathlib import Path
from threading import Thread
import argparse
import glob

import cv2
import numpy as np


import datetime
import time

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='NorthGate', help='source')  # input file/folder, 0 for webcam

    opt = parser.parse_args()
    
    print(opt)

    source = opt.source
    path = str(Path(source))

    while True:
        files = os.listdir(path)

        print("Current files's count:",len( files ))

        if len( files ) > 3000:
            files.sort(key=lambda x: int(x.split('.')[0]))
            print('files len over 3000:', len(files) )
            del_list = files[:1000]
            #print(del_list)
            for filename in del_list:
                if os.path.exists(path + '/' + filename):
                    os.remove(path + '/' + filename)
                    #print('delete image:',path + '/' + filename)

        time.sleep(10)