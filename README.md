# Architecture

## License Plate Recognition System
![](https://nol.cs.nctu.edu.tw:234/iTraffic/Simon/raw/763cedc6e4c8d79b363a8c07fd008fc8ef51cee9/Images/%E8%BB%8A%E7%89%8C%E8%BE%A8%E8%AD%98%E6%B5%81%E7%A8%8B%E5%9C%96.PNG)

## LPR with Deep Learning
![](https://nol.cs.nctu.edu.tw:234/iTraffic/Simon/raw/a5cacf48701bd1436f5a43a2e2d1b62564a8449c/Images/LPR_with_Deep_Learning.PNG)

# Environments

## Environments Setting
There are two ways to set up the environment, because the use environment uses the yolov3 environment provided by ultralytics in the Docker Hub on the Internet, but the author has continued to optimize the environment, and there are many differences between the environment and operation when I used it at that time.
If you want to use the new version of the original author's environment, you can refer to the Method 1; if you want to reproduce the environment I use, you can refer to the Method 2.

### Method 1
- Step 1 - Docker Image

Use the yolov3 environment provided by ultralytics in Docker Hub

https://hub.docker.com/r/ultralytics/yolov3

Download dcker image
```
docker pull ultralytics/yolov3
```
Create container

```
sudo docker run --gpus all --ipc=host --net=host -v 本機端共用資料夾路徑:/workspace/yolov3 -it --name Container名稱 ultralytics/yolov3 bash
```

- Step 2 - Install

After opening and entering container, go to the webpage and find the title "Tutorials"->"Train Custom Data ", and follow the instructions

https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data

```
$ git clone https://github.com/ultralytics/yolov3  # clone repo
$ cd yolov3
$ pip install -r requirements.txt  # install dependencies
```

### Method 2
- Step 1 - Docker Image

Use what i made's Donker Image to create container
Donker Image Name: simon/yolov3-20210319:ver1.0
Donker Image Location: in 140.113.208.104 server
Donker Image ID: c01633076fa7c6

```
sudo docker run --gpus all --ipc=host --net=host -v 本機端共用資料夾路徑:/workspace/yolov3 -it --name Container名稱 simon/yolov3-20210319:ver1.0 bash
```

- Step 2 - Copy Data

Download all files in the Code folder and put them under the /workspace/yolov3 folder

https://nol.cs.nctu.edu.tw:234/iTraffic/Simon/tree/master/Code