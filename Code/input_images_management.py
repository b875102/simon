import os
import time
import argparse
import datetime
import sys

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='NorthGate', help='source')  # input file/folder, 0 for webcam

    opt = parser.parse_args()
    
    print(opt)

    source = opt.source
    path = str(Path(source))

    #intput_path = '/workspace/yolov3/yolov3/' + path
    #print("intput_path: ",intput_path)

    while True:
        files = os.listdir(path)

        print("Current files's count:",len( files ))
        sys.stdout.flush()

        if len( files ) > 1000:
            files.sort(key=lambda x: int(x.split('.')[0]))
            print('files len over 1000:', len(files) )
            del_list = files[:600]
            #print(del_list)
            for filename in del_list:
                if os.path.exists(path + '/' + filename):
                    os.remove(path + '/' + filename)
                    #print('delete image:',path + '/' + filename)

        time.sleep(10)