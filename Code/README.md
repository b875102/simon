- <h4>Training

Execute the following command under the yolov3 folder:

1. --cfg    : cfg file used during training
2. --data : the .data file to train
3. --weights  : pretrained model
4. see the train.py file for other parameters

Please modify the path of yolov3.cfg, obj.data and darknet53.conv.74 to your place
The dataset of this research is placed in the following path: https://drive.google.com/drive/folders/1PFfBDMD40GNB46zft-zvGTRM0_9nObIB

```
python3 train.py --cfg /workspace/yolov3/character/yolov3.cfg --data /workspace/yolov3/character/cfg/obj.data --weights /workspace/yolov3/darknet53.conv.74 --batch-size 12
```
- <h4>Inference

- 1. detect.py ，detect object( ex.license plate )

Execute the following command under the yolov3 folder:


1. --source : input( video, images ,or all files under the folder )
2. --cfg    : cfg file used during training
3. --weights : training weight
4. --names  : class name
5. see the detect.py file for other parameters


```
python3 detect.py --source /workspace/yolov3/test_video/GH060927.MP4 --cfg /workspace/yolov3/vehicle/cfg/yolov3.cfg --weight /workspace/yolov3/vehicle/cfg/weights-1103/best.pt --names /workspace/yolov3/vehicle/cfg/obj.names
```


- 2. detect-1113.py ，detect the license plate first and then identify the license plate number

Execute the following command under the yolov3 folder:

1. --source : input( video, images ,or all files under the folder )
2. --cfg    : cfg file used during license plate training
3. --weights : license plate training weight
4. --names  : license plate class name
5. --lpcfg    : cfg file used during character training
6. --lpweights : character training weight
7. --lpnames  : character class name
8. see the detect-1113.py file for other parameters

```
python3 detect-1113.py --source data/test_image/ --lpcfg /workspace/yolov3/character/cfg/yolov3-spp.cfg --lpweights /workspace/yolov3/character/cfg/final/best.pt --lpnames /workspace/yolov3/character/cfg/obj.names --cfg /workspace/yolov3/vehicle/cfg/yolov3.cfg --weights /workspace/yolov3/vehicle/cfg/weights-1103/best.pt --names /workspace/yolov3/vehicle/cfg/obj.names
```

- 3. detect-cropLP ，after detect the license plate, crop the license plate image, and resize to a height of 200 pixels according to the ratio, and then output

Execute the following command under the yolov3 folder:

1. --source : input( video, images ,or all files under the folder )
2. --cfg    : cfg file used during license plate training
3. --weights : license plate training weight
4. --names  : license plate class name

```
python3 detect-cropLP.py --source data/test_image/ --lpcfg /workspace/yolov3/character/cfg/yolov3-spp.cfg --lpweights /workspace/yolov3/character/cfg/final/best.pt --lpnames /workspace/yolov3/character/cfg/obj.names --cfg /workspace/yolov3/vehicle/cfg/yolov3.cfg --weights /workspace/yolov3/vehicle/cfg/weights-1103/best.pt --names /workspace/yolov3/vehicle/cfg/obj.names
```

- 4. detect-multi-vg-fps-20210125 ，license plate recognition can only run on pre-recorded videos, and finally the license plate image and the recognition result video will be output

Execute the following command under the yolov3 folder:

1. --source : the folder name of the file to be predicted, must be given
2. --cfg    : cfg file used during license plate training
3. --weights : license plate training weight
4. --names  : license plate class name
5. --lpcfg    : cfg file used during character training
6. --lpweights : character training weight
7. --lpnames  : character class name
8. --virtualgate : virtual gate XXX-vg.txt
8. see the detect-images-20210317.py file for other parameters

```
python3 detect-multi-vg-fps-20210125.py --source NorthGate --lpcfg /workspace/yolov3/character/cfg/yolov3-spp.cfg --lpweights /workspace/yolov3/character/cfg/final/best.pt --lpnames /workspace/yolov3/character/cfg/obj.names --cfg /workspace/yolov3/vehicle/cfg/yolov3.cfg --weights /workspace/yolov3/vehicle/cfg/weights-1103/best.pt --names /workspace/yolov3/vehicle/cfg/obj.names
```
給定 source 資料夾名稱後，程式會從 source 路徑底下逐一讀取影片做車牌辨識。同時也會從 --virtualgate參數路徑下，與影片位置上一層的資料夾名稱相同的資料夾內讀取 資料夾名稱-vg.txt檔案。
當確認車牌結果時，會將影像輸出至與detect-images-20210317.py檔案同路徑下的 output/影片位置上一層的資料夾名稱 底下。

After give the source folder name, the program will read the videos one by one from the source path for license plate recognition. 
At the same time, from the path of the --virtualgate parameter, the folder name-vg.txt file in the folder with the same name as the folder above the video location will be read.
When the license plate result is confirmed, the image will be output to the folder name one level above the output/video location in the same path as the detect-images-20210317.py file.

- 5. detect-images-20210317 ，the license plate recognition system code of this research, the process is as the license plate recognition flowchart in the above chapter

Execute the following command under the yolov3 folder:

1. --source : the folder name of the file to be predicted, must be given
2. --cfg    : cfg file used during license plate training
3. --weights : license plate training weight
4. --names  : license plate class name
5. --lpcfg    : cfg file used during character training
6. --lpweights : character training weight
7. --lpnames  : character class name
8. see the detect-images-20210317.py file for other parameters


```
python3 detect-images-20210317.py --source NorthGate --lpcfg /workspace/yolov3/character/cfg/yolov3-spp.cfg --lpweights /workspace/yolov3/character/cfg/final/best.pt --lpnames /workspace/yolov3/character/cfg/obj.names --cfg /workspace/yolov3/vehicle/cfg/yolov3.cfg --weights /workspace/yolov3/vehicle/cfg/weights-1103/best.pt --names /workspace/yolov3/vehicle/cfg/obj.names
```
給定 source 資料夾名稱後，程式會從與detect-images-20210317.py檔案同路徑下的 input/source資料夾名稱 底下，從檔名由 1 開始往上計數的png檔逐一讀取影像做車牌辨識(模擬串流下載的影像名稱，讀取方式可在 utils/datasets.py 的 class LoadImages 中修改)。
同時也會從與detect-images-20210317.py檔案同路徑下的 data/virtual_gate/source資料夾名稱 底下，讀取 source資料夾名稱-vg.txt 的 virtual gate 設定檔。
當確認車牌結果時，會將影像輸出至與detect-images-20210317.py檔案同路徑下的 output/source資料夾名稱 底下。

Given the name of the source folder, the program will read the images from the input/source folder name in the same path as the detect-images-20210317.py file, and the png files with the file name starting from 1 and counting upwards for license plate recognition ( The image name of the simulated streaming download, the reading method can be modified in class LoadImages in utils/datasets.py).
At the same time, it will also read the virtual gate configuration file of the source folder name-vg.txt from the data/virtual_gate/source folder name under the same path as the detect-images-20210317.py file.
When the license plate result is confirmed, the image will be output to the output/source folder name under the same path as the detect-images-20210317.py file.




