import os
import time
import argparse
import pexpect

import signal
import sys
import subprocess

from pathlib import Path

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    # kill process
    time.sleep(3)

    #os.killpg(os.getpgid(test_process.pid),signal.SIGTERM)
    os.killpg(os.getpgid(send_result_images_process.pid),signal.SIGTERM)
    os.killpg(os.getpgid(ffmpeg_process.pid),signal.SIGTERM)
    os.killpg(os.getpgid(input_images_management_process.pid),signal.SIGTERM)
    #os.killpg(os.getpgid(docker_process.pid),signal.SIGTERM)
    
    sys.exit(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='NorthGate', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--web', type=str, default='NorthGate', help='source')  # input file/folder, 0 for webcam

    opt = parser.parse_args()
    
    print(opt)

    source = opt.source
    web = '"' + "rtmp://127.0.0.1:1935/live/livestream" + '"'
    
    #####################################################################################
    # 執行 ffmpeg 指令
    #stream_image_output_path = '"' + "/nol/simon/yolov3/" + source + "/%d.png" + '"'
    stream_image_output_path = '"' + source + "/%d.png" + '"'
    image_scale = '"' + "scale='w=960:h=540',fps=30" + '"'

    #ffmpeg_cmdline = "gnome-terminal -e 'bash -c \"ffmpeg -i %s -vf %s %s\"'" %(web, image_scale, stream_image_output_path)
    ffmpeg_cmdline = "ffmpeg -i %s -vf %s %s" %(web, image_scale, stream_image_output_path)
    print('ffmpeg_cmdline:',ffmpeg_cmdline)

    ffmpeg_process = subprocess.Popen(ffmpeg_cmdline,stdout=subprocess.PIPE, shell=True)
    #os.system(ffmpeg_cmdline)
    '''
    #test 
    test_cmd = "./LPR_Main.sh"
    test_process = subprocess.Popen(test_cmd,stdout=subprocess.PIPE, shell=True)
    '''

    #####################################################################################
    # 執行傳送影像.py檔
    #send_result_images_cmdline = "gnome-terminal -e 'bash -c \"python3 /nol/simon/yolov3/send_result_images.py --source %s\"'" %(source)
    #send_result_images_cmdline = "python3 /nol/simon/yolov3/send_result_images.py --source %s" %(source)
    send_result_images_cmdline = "python3 send_result_images.py --source %s" %(source)
    print('send_result_images_cmdline:',send_result_images_cmdline)

    send_result_images_process = subprocess.Popen(send_result_images_cmdline,stdout=subprocess.PIPE, shell=True)
    #os.system(send_result_images_cmdline)

    #####################################################################################
    # 執行串流影像管理.py檔
    #input_images_management_cmdline = "gnome-terminal -e 'bash -c \"python3 /nol/simon/yolov3/input_images_management.py --source %s\"'" %(source)
    #input_images_management_cmdline = "python3 /nol/simon/yolov3/input_images_management.py --source %s" %(source)
    input_images_management_cmdline = "python3 input_images_management.py --source %s" %(source)
    print('input_images_management_cmdline:',input_images_management_cmdline)
    
    input_images_management_process = subprocess.Popen(input_images_management_cmdline,stdout=subprocess.PIPE, shell=True)
    #os.system(input_images_management_cmdline)


    time.sleep(3)

    #####################################################################################
    # 執行 docker detect.py 檔
    #docker_cmdline = "gnome-terminal -e 'bash -c \"sudo docker exec -it 0315 bash \"'"
    docker_cmdline = "python3 detect-images-20210317.py --source %s" %(source)
    print('docker_cmdline:',docker_cmdline)
    os.system(docker_cmdline)
    #docker_process = subprocess.Popen(docker_cmdline,stdout=subprocess.PIPE, shell=True)
    #os.system(docker_cmdline)

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C')
    signal.pause()
