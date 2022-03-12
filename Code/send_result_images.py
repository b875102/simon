import os
import time
import argparse
import pexpect
import sys

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='NorthGate', help='source')  # input file/folder, 0 for webcam

    opt = parser.parse_args()
    
    print(opt)

    source = opt.source

    intput_path = 'output/' + str(Path(source))
    #intput_path = '/workspace/yolov3/yolov3/output/' + str(Path(source))
    #intput_path = '/nol/simon/yolov3/output/' + str(Path(source))
    output_path = 'output-test/' + str(Path(source))

    print("intput_path:",intput_path)
    print("output_path:",output_path)

    total_update_count = 0

    while True:
        files = os.listdir(intput_path)

        print("Total_update_count:",total_update_count)
        #sys.stdout.flush()
        
        if len( files ) > 0:   
            for filename in files:
                print("input files:", filename )

                file_full_path = intput_path + '/' + filename

                if os.path.exists(file_full_path):
                    cmdline = "scp %s iscocm@140.113.208.118:~/Downloads/%s" %(file_full_path,output_path)
                    print(cmdline)
                 
                    try:   
                        child = pexpect.spawn(cmdline)
                        r= child.expect("password:")
                        child.sendline("iscom")
                        child.read()

                        print('Upload Success!:',file_full_path)
                        total_update_count += 1
                    except:
                        print('Upload faild!:',file_full_path)

                    os.remove(file_full_path)
        
        time.sleep(1)