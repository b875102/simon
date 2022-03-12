import argparse
import cv2

from math import hypot
import sys
import numpy as np
import time
import ffmpeg
from datetime import datetime
import difflib

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import threading

'''
output_result_image_list = []
isProcessEnd = False
'''

###############################################################
class_index = { '0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,\
                'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18, 'J':19,\
                'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27, 'S':28, 'T':29,\
                'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35 }

###############################################
class_Road_Side = { 0:'', 1:'-Left', 2:'-Right' }

###############################################
###############################################
class Character():
    def __init__(self,name,location):
        self.name = str(name)
        self.location_X = int( ( location[0] + location[2] ) / 2 ) ### location[0] : x1，[2] : x2
        self.location_Y = int( ( location[1] + location[3] ) / 2 ) ### location[1] : y1，[3] : y2
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
###############################################
class Vehicle():
    def __init__( self, ID, BoundingBox, CenterPoint, IsMark, LPNumber={}, Distance = 0, UnUpdateCount = 0, Image = None, SavePath = '', RoadSideIndex = '' ):
        self.id = ID
        self.bbox = BoundingBox
        self.cp = CenterPoint
        self.isMark = IsMark
        self.number_dict = LPNumber
        self.distance = Distance
        self.unUpdateCount = UnUpdateCount
        self.outputSavePath = SavePath
        self.roadSideIndex = RoadSideIndex     
        self.image = Image
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    
    def __repr__(self):
        return str(self)
###############################################
def getListUnuseID( last_list, current_list ):
    if len(last_list) == 0 and len(current_list) == 0:
        return 1
    
    if len(last_list) == 0:
        for i in range( 1,sys.maxsize):
            repeat = False
            for current in current_list:
                if i == current.id:
                    repeat = True
            if repeat == False:
                return i
    elif len(current_list) == 0:
        for i in range( 1,sys.maxsize):
            repeat = False
            for last in last_list:
                if i == last.id:
                    repeat = True
            if repeat == False:
                return i
    else:
        for i in range( 1,sys.maxsize):
            repeat = False
            for last in last_list:
                if i == last.id:
                    repeat = True
                else:
                    for current in current_list:
                        if i == current.id:
                            repeat = True
            if repeat == False:
                return i
###############################################
def count_Object(centroid_last, centroid_now):
    # Computes distance between two point
    def distance(pt_now, pt_last):      
        return hypot(pt_now[0]-pt_last[0], pt_now[1]-pt_last[1])

    def license_plate_imilarity(string_now,string_last):
        return difflib.SequenceMatcher(None, string_now, string_last).quick_ratio()
  
    for cent_last in centroid_last:
        dist = 0.0      
        distances = []
        
        for cent_now in centroid_now:
            dist = distance(cent_now.cp, cent_last.cp)
            #print('cent_now.cp:{0}, cent_last.cp:{1}, dist:{2}'.format(cent_now.cp,cent_last.cp,dist))
            distances.append(dist)
        
        if len(distances) != 0:
            min_dis_index = distances.index(min(distances))
            #print('distances',distances, min_dis_index)

            if len(cent_last.number_dict) != 0:          
                last_lp_number = max(cent_last.number_dict, key=cent_last.number_dict.get)
                now_lp_number = max(centroid_now[min_dis_index].number_dict, key=centroid_now[min_dis_index].number_dict.get)
                #print('last_lp_number:',last_lp_number)
                #print('now_lp_number:',now_lp_number)
                lp_number_ratio = license_plate_imilarity(now_lp_number,last_lp_number)
                #print('lp_number_ratio',lp_number_ratio)
            else:
                lp_number_ratio = 0

            if lp_number_ratio > 0.8:
                centroid_now[min_dis_index].id = cent_last.id
                centroid_now[min_dis_index].isMark = cent_last.isMark
                centroid_now[min_dis_index].distance = ( distances[min_dis_index] + cent_last.distance ) / 2
            if lp_number_ratio == 0 and distances[min_dis_index] < 108:
                centroid_now[min_dis_index].id = cent_last.id
                centroid_now[min_dis_index].isMark = cent_last.isMark
                centroid_now[min_dis_index].distance = ( distances[min_dis_index] + cent_last.distance ) / 2
    return centroid_now
###############################################
def update_Centroid_Last_Data(centroid_last, centroid_now, fps_count):
    def HandleLastData( data_last, data_now ):
        data_last.id = data_now.id
        data_last.bbox = data_now.bbox
        data_last.cp = data_now.cp    
        data_last.distance = data_now.distance
        data_last.unUpdateCount = 0

        if data_last.isMark == False:
            data_last.isMark = data_now.isMark

        if data_last.isMark == True:
            #data_last.unUpdateCount += 1 # 開始計數 10張 frame

            if ( len( data_last.number_dict ) == 0 ):
                data_last.number_dict = data_now.number_dict
            else:
                for key, value in data_now.number_dict.items():
                    if key in data_last.number_dict.keys():    
                        data_last.number_dict[ key ] += 1
                    else:
                        data_last.number_dict.setdefault( key, 1 )
        else:
            data_last.number_dict = data_now.number_dict

        if data_now.roadSideIndex != '':
            data_last.roadSideIndex = data_now.roadSideIndex
        
        if np.max(data_now.image) is not None:
            data_last.image = data_now.image
            data_last.outputSavePath = data_now.outputSavePath

        return data_last

    # 把 centroid_now 更新到 centroid_last, cent_last 未被更新到的 unUpdateCount + 1
    for last_idx in range( len( centroid_last ) ):   
        cent_last = centroid_last[ last_idx ]
        isUpdate = False
        for now_idx in range( len( centroid_now ) ):
            cent_now = centroid_now[ now_idx ]

            if cent_last.id == cent_now.id:
                isUpdate = True
                centroid_last[ last_idx ] = HandleLastData( centroid_last[ last_idx ], centroid_now[ now_idx ] )
        if isUpdate == False:
           centroid_last[ last_idx ].unUpdateCount += 1


    # 把 centroid_now ID 沒有重複的更新到 centroid_last
    temp_list = centroid_last

    for cent_now in centroid_now:
        isRepeat = False
        for cent_last in centroid_last:
            if cent_last.id == cent_now.id:
                isRepeat = True
        if isRepeat == False:
            temp_list.append(cent_now)

    # 移除 unUpdateCount 超過 100 次( 即累積100張 frame都未出現 )的車牌
    for i in range(len(temp_list)-1,-1,-1):
        # 累積 10 筆資料就輸出影像
        number_count = 0
        for key, value in temp_list[i].number_dict.items():
            number_count += value
        #if number_count >= 3 and number_count <= 10:
        print(number_count)
        if number_count == 5:
            print('temp_list[i]:: ', temp_list[i].id)
            if np.max(temp_list[i].image) is not None:
            #if temp_list[i].image != None:
                # 存圖片出去
                lp_number = max(temp_list[i].number_dict, key=temp_list[i].number_dict.get)
                #path = temp_list[i].outputSavePath + "-" + lp_number + "-" + str(number_count) + ".png"
                path = temp_list[i].outputSavePath + "-" + lp_number + ".png"
                print('save_path: ', path )

                # 先要獲取鎖:
                #lock.acquire()
                #output_result_image_list.append( [ path, temp_list[i].image ] )
                #lock.release()

                cv2.imwrite( path, temp_list[i].image )

        # 超過 600 筆資料就移除紀錄
        if temp_list[i].unUpdateCount > fps_count*0.5:
            temp_list.pop(i)
    
    #print('cent_now',centroid_now)
    #print('tem_list',temp_list)
    return temp_list
###############################################
##       輸出車牌影像
'''
def output_image_result():
    global output_result_image_list
    global isProcessEnd
    
    while not isProcessEnd:
        #print(ouput_list)
        if len(output_result_image_list) > 0:
            
            #lock.acquire()
            
            # 輸出影像的路徑
            output_result_image_path = output_result_image_list[0][0]
             # 輸出影像的路徑
            output_result_image = output_result_image_list[0][1]

            print('save_path: ', output_result_image_path)

            cv2.imwrite( output_result_image_path, output_result_image )
            
            #清除output_result_image_list輸出的影像
            output_result_image_list.pop(0)
            print('輸出圖片完成')
            #lock.release()
'''
###############################################
def FilterLicensePlateCandidate( candidate ):
    plate_list = []
    
    for i in range( len(candidate) ):
        plate_list.append(candidate[i][0])
        
    return plate_list
###############################################
def LicensePlateRule( LPstring ):
    length = len( LPstring )
    
    # 車牌長度超過7 或小於4 代表有問題
    if length > 7 or length < 4:
        return False
    
    # 計算 A-Z 數量
    char_count = 0
    for i in range( len(LPstring) ):
        if class_index[ LPstring[ i ] ] > 9:
            char_count += 1

    if length == 7:
        if char_count == 3:
            return True
        else:
            return False
    else:
        if char_count <= 3 and char_count >= 1: # TBC160 taxi
            return True
        else:
            return False
###############################################################
def LoadVirtualGate( path ):
    vgs_data = np.loadtxt('%s' % path, dtype=int).reshape(-1,5)
    
    #scan_direction = []
    #areaRange = []
    vg_pt = []
    road_side = []

    #print(path)
    #print(vgs_data)
    #print(len(vgs_data))
    for vgs in vgs_data:
        print(vgs)
        if len(vgs)==5:
            p1 = [vgs[0],vgs[1]] # 左上
            p2 = [vgs[2],vgs[3]] # 右下
            #areaRange.append(vgs[4])
            #scan_direction.append(vgs[5]) # 0 是左右方向截圖, 1 是上下方向截圖
            road_side.append(vgs[4]) # 1 是左線道, 2 是右線道
        else:
            p1 = [0,0] # 左上
            p2 = [0,0] # 右下
            #areaRange.append(0)
            #scan_direction.append(0) # 0 是左右方向截圖, 1 是上下方向截圖
            road_side.append(0) # 1 是左線道, 2 是右線道

        vg_pt.append([p1, p2])
    
    print('VirtualGate:{0},Road_side:{1}'.format(vg_pt,road_side))

    return vg_pt, road_side

###############################################################
def WriteFrameTXTData( path, data_list ):
    fp = open( path, "w")
    for data in data_list:
        for item in data:
            fp.write(str(item)+' ')
        fp.write('\n')
    
    #關閉檔案
    fp.close()  
    return True

###############################################################
def IsIntersec(p1,p2,p3,p4): #判断两线段是否相交
    def cross(p1,p2,p3):#跨立实验
        x1=p2[0]-p1[0]
        y1=p2[1]-p1[1]
        x2=p3[0]-p1[0]
        y2=p3[1]-p1[1]
        return x1*y2-x2*y1   

    D = False
    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if( max(p1[0],p2[0])>=min(p3[0],p4[0])    #矩形1最右端大于矩形2最左端
    and max(p3[0],p4[0])>=min(p1[0],p2[0])    #矩形2最右端大于矩形最左端
    and max(p1[1],p2[1])>=min(p3[1],p4[1])   #矩形1最高端大于矩形最低端
    and max(p3[1],p4[1])>=min(p1[1],p2[1]) ):  #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0
           and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            D = True

    return D

###############################################################

def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    lpweights = opt.lpweights

    setting_fps = opt.FPS

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    #if os.path.exists(out):
    #    shutil.rmtree(out)  # delete output folder
    #os.makedirs(out)  # make new output folder
    if not os.path.exists(out):
        os.makedirs(out)

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    lpmodel = Darknet(opt.lpcfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    attempt_download(lpweights)
    if lpweights.endswith('.pt'):  # pytorch format
        lpmodel.load_state_dict(torch.load(lpweights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(lpmodel, lpweights)

    # Eval mode
    model.to(device).eval()

    lpmodel.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
        lpmodel.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        #save_img = True
        image_source = 'input/' + source
        dataset = LoadImages(image_source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    lpnames = load_classes(opt.lpnames)
    lpcolors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(lpnames))]

    # Run inference
    print('total len: ', len(dataset))
    dataset_count = 0
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    _ = lpmodel(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    ###############################################
    last_frame_data = []

    # set virtual gate and detect range
    p1 = [1600, 10] # 上
    p2 = [1600, 1070] # 下
    
    vg_pt = [[p1, p2]]
    calculateRange = [250]
    scan_direction = [0]
    road_side = [0]

    cur_video_path = ''

    if setting_fps == 0:
        FPS_Enable = False
    else:
        FPS_Enable = True

    frame_rate = 0
    last_virtualgate = ''
    image_save_path = out

    current_frame_data = []
    video_FPS = 30

    each_frame_data = []
    video_name = ''

    # 重新載入新的 virtualgate 檔案
    virtualgate = opt.virtualgate + '/' + source +'-vg.txt'  
    print("virtualgate:",virtualgate) 
    # Load virtual gate
    vg_pt, road_side = LoadVirtualGate( virtualgate )  

    # 建立 out 新資料夾
    image_save_path = str(Path(out) / source)
   
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    

    ###############################################
    for path, img, im0s, vid_cap in dataset:
        # Inference
        t1 = torch_utils.time_synchronized()
        # 重製當下 frame 資料
        current_frame_data = []
        ###############################################
        if dataset.mode == 'video':
            # 重製當下 frame 資料
            current_frame_data = []
            if cur_video_path != path:
                cur_video_path = path
                # 重製 dataset_count
                dataset_count = 0
                # 讀取影片的時長
                #video_info = get_video_info(cur_video_path)
                # 獲取影片資訊 - 開始時間
                #video_creation_time = get_video_create_time( video_info )
                # 獲取影片資訊 - FPS
                #video_FPS = get_video_FPS( video_info )
                video_FPS = 30
                # 影像總張數
                #total_duration = video_info['duration']

                if FPS_Enable:
                    if setting_fps > video_FPS or setting_fps < 0:
                        setting_fps = video_FPS

                    frame_rate = int( video_FPS / setting_fps )

                #if dataset_count == 0:
                video_name = path.split('/')[-1].split('.')[0]
                new_video_virtualgate = path.split('/')[-2]
                print('last_virtualgate:{0},new:{1}'.format(last_virtualgate,new_video_virtualgate))
                if new_video_virtualgate != last_virtualgate:
                     # 建立 out 新資料夾
                    image_save_path = str(Path(out) / new_video_virtualgate)

                    last_virtualgate = new_video_virtualgate

                   
                    if not os.path.exists(image_save_path):
                        os.makedirs(image_save_path)

                    # 重新載入新的 virtualgate 檔案
                    virtualgate = opt.virtualgate + '/' + last_virtualgate +'-vg.txt'   
                    # Load virtual gate
                    vg_pt, road_side = LoadVirtualGate( virtualgate )  

                # Reset last_frame_data
                last_frame_data = []               
                #print( 'Reset last_frame_data: ',last_frame_data )

                print( 'setting_fps: ',setting_fps )
        ###############################################

        dataset_count+=1

        if FPS_Enable and int( dataset_count % frame_rate ) != 0:
            print('dataset_count Skip: ',dataset_count)
            # Save results (image with detections)
            if save_img and not webcam:

                p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)

                #print('save_img')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
            continue

        #print(dataset_count)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        #t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        #t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # list
        lp_list = []

        # Process detections
        ####################
        # 找車牌
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    cls1 = cls
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

		    ###############################################################
                    # handle license plate class
                    #if names[int(cls)] == 'license plate':

                    ###############################################################
                    # x1 : int(xyxy[ 0 ]), y1 : int(xyxy[ 1 ]), x2 : int(xyxy[ 2 ]), y2 : int(xyxy[ 3 ])
                    bbox = [ int(xyxy[ 0 ]),int(xyxy[ 1 ]),int(xyxy[ 2 ]),int(xyxy[ 3 ]) ]
                    ###############################################################
                    final_LP = ''
                    # crop license image
                    print(int(xyxy[ 0 ]) , int(xyxy[ 1 ]), int(xyxy[ 2 ]) , int(xyxy[ 3 ]) )
                    lp_im0 = im0[ int(xyxy[ 1 ]) : int(xyxy[ 3 ]), int(xyxy[ 0 ]) : int(xyxy[ 2 ]) ]
                	# resize image height to 200
                    rate = 200 / ( int(xyxy[ 3 ]) - int(xyxy[ 1 ]) )
                    lp_im0 = cv2.resize(lp_im0, ( int( ( int(xyxy[ 2 ]) - int(xyxy[ 0 ]) ) * rate ), 200), interpolation=cv2.INTER_CUBIC)
                	
    	            # Padded resize
                    lp_img = letterbox(lp_im0, imgsz)[0]
                    lp_img = lp_img[:, :, ::-1].transpose(2 , 0, 1)
                    lp_img = np.ascontiguousarray(lp_img)

                    lp_img = torch.from_numpy(lp_img).to(device)
                    lp_img = lp_img.half() if half else lp_img.float()  # uint8 to fp16/32
                    lp_img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if lp_img.ndimension() == 3:
                        lp_img = lp_img.unsqueeze(0)


		            # Inference
                    lp_pred = lpmodel(lp_img, augment=opt.augment)[0]

		            # to float
                    if half:
                        lp_pred = lp_pred.float()

		            # Apply NMS
                    lp_pred = non_max_suppression(lp_pred, opt.conf_thres, opt.iou_thres, multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
                    ####################
                    # 找車牌字元
                    for lp_i, lp_det in enumerate(lp_pred):  # detections for image lp_i
                        if lp_det is not None and len(lp_det):
		                    # Rescale boxes from imgsz to im0 size
                            lp_det[:, :4] = scale_coords(lp_img.shape[2:], lp_det[:, :4], lp_im0.shape).round()
                            #lp_det[:, :4] = scale_coords(lp_im0.shape[2:], lp_det[:, :4], lp_im0.shape).round()

                            h = lp_im0.shape[0]

                            y_min_range = h * 0.25
                            y_max_range = h * 0.75

                            license_plate_candidate = []
                            char_len = len(lp_det)

			                # Write results
                            for *lp_xyxy, lp_conf, cls in reversed(lp_det):
                                locate = ( int(lp_xyxy[0]), int(lp_xyxy[1]), int(lp_xyxy[2]), int(lp_xyxy[3]) )
        
                                char = Character( lpnames[int(cls)], locate )
                                #print(int(cls))
                                if char.location_Y > y_max_range or char.location_Y < y_min_range:
                                    char_len -= 1
                                    continue

                                license_plate_candidate.append( ( char.name, char.location_X, char.location_Y ) )

			                # 用 x 座標由左到右排列
		                    #print(license_plate_candidate)
                            license_plate_candidate = sorted(license_plate_candidate, key = lambda s: s[ 1 ])
		            
                            #print(license_plate_candidate)
		        
		                    # 過濾車牌字元
                            license_plate = FilterLicensePlateCandidate( license_plate_candidate )
                            #print(license_plate)
		                    
                            # list 轉乘 string
                            final_LP = '%s'*len(license_plate) % tuple(license_plate)
                            #print('final_LP:',final_LP)
                            #final_LP = LicensePlateRule( license_plate )
                    ###############################################################
                    if dataset.mode == 'video' or dataset.mode == 'images':
                        # 若處於熱區範圍，則與上一個frame的車輛進行比對，計算該車中心點與上一個frame所有車輛的中心點距離最短是那一台。
                        #if isInTrackingRegion_result == True:
                        if LicensePlateRule( final_LP ) == True:
                            ctp = [ int( ( bbox[0] + bbox[2] ) / 2 ), int( ( bbox[1] + bbox[3] ) / 2 ) ]

                            new_id = getListUnuseID( last_frame_data, current_frame_data )
                            #print('new_id',new_id,final_LP)
                            LP_dict = {}
                            LP_dict.setdefault( final_LP, 1 )
                            new_vehicle = Vehicle( new_id, bbox, ctp, False, LP_dict )
                            current_frame_data.append(new_vehicle)
                            #else:
                            #    print('LicensePlateRule == False:',final_LP)
                        
                    lp_list.append([xyxy,final_LP])    
            ###############################################################
            if dataset.mode == 'video' or dataset.mode == 'images':
                ####################
                # 比對前後資料
                current_frame_data = count_Object( last_frame_data, current_frame_data )

                ####################
                # 判斷是否跨越 VG
                for last_data in last_frame_data:
                    for current_data in current_frame_data:
                        if last_data.id == current_data.id:
                            LEFT = False
                            RIGHT = False
                            UP = False
                            DOWN = False

                            if last_data.isMark == True:
                                continue

                            for vg_idx in range( len( vg_pt ) ):
                                vg_data = vg_pt[vg_idx]
                                #scan_direction_data = scan_direction[vg_idx]
                                road_side_data = road_side[vg_idx]

                                #print(last_data.cp[1], vg_pt[0][1], current_data.cp[1], vg_pt[0][1])
                                #print(vg_data[0], vg_data[1], current_data.cp, last_data.cp)
                                intersec = IsIntersec(vg_data[0],vg_data[1],current_data.cp,last_data.cp)
                                if( intersec  == True):
                                    print( 'intersec == TRUE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')

                                    current_data.isMark = True
                                    current_data.image = im0

                                    #print('road_side_data: ', road_side_data,class_Road_Side[road_side_data])
                                    #aaa = 
                                    current_data.roadSideIndex = class_Road_Side[road_side_data]
                                    # 將選擇時間轉換成經過的時間( 單位:秒 )
                                    #occur_sec = int( cur_video_frame_cnt / video_FPS )
                                    #print('Take frame count : {0}, Occur second : {1}'.format( cur_video_frame_cnt, occur_sec ) )

                                    # 將經過的時間更新到現實時間
                                    new_utc_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")#video_creation_time + datetime.timedelta( seconds = occur_sec )
                                    print( 'new_utc_time : ', new_utc_time )
                                    picture_name =  str(new_utc_time).replace(" ","-").replace(":","_")
                                    picture_name = picture_name + class_Road_Side[road_side_data]          
                                    #picture_name =  str(new_utc_time).replace(" ","-") + "-" + current_data.number + ".png"
                                    output_save_path = image_save_path + '/' + picture_name
                                    print('output_save_path: ',output_save_path)
                                    current_data.outputSavePath = output_save_path

                                    # print(current_data)
                                    # os._exit(0)
                                    break

                last_frame_data = update_Centroid_Last_Data(last_frame_data, current_frame_data, video_FPS )                   
                #last_frame_data = current_frame_data
            # Draw result
            if save_img:
                for lp_index in lp_list:
                    box = [ int(lp_index[ 0 ][ 0 ]),int(lp_index[ 0 ][ 1 ]),int(lp_index[ 0 ][ 2 ]),int(lp_index[ 0 ][ 3 ]) ]
                    lp_name = lp_index[1]

                    for last_data in last_frame_data:
                        if last_data.bbox == box:
                            print('lp_name:{0}, ast_data.roadSideIndex:{1}'.format(lp_name,last_data.roadSideIndex))
                            print('last_data.bbox:',last_data.bbox)
                            lp_name = lp_name + last_data.roadSideIndex
                            break
                    
                    #print('lp_index[0]: ',lp_index[0])
                    plot_one_box(lp_index[0], im0, label=lp_name, color=colors[0])

            ###############################################################
            # Print time (inference + NMS)
            t2 = torch_utils.time_synchronized()
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    print('Stop Iteration!!')
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                #print('save_img')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    '''
    while len(output_result_image_list) != 0: 
        isProcessEnd = False
    isProcessEnd = True
    output_image_result_thread.join()
    '''

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/workspace/yolov3/vehicle/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='/workspace/yolov3/vehicle/cfg/obj.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='/workspace/yolov3/vehicle/cfg/weights-1210/best.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--lpcfg', type=str, default='/workspace/yolov3/character/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--lpnames', type=str, default='/workspace/yolov3/character/cfg/obj.names', help='*.names path')
    parser.add_argument('--lpweights', type=str, default='/workspace/yolov3/character/cfg/weights-20210127-ep300/last.pt', help='weights path')
    parser.add_argument('--virtualgate', type=str, default='data/virtual_gate', help='read xxx-vg.txt')
    parser.add_argument('--FPS', type=int, default='0', help='take image frame rate')
    parser.add_argument('--save-result', action='store_true', help='save results to *.png or *.mp4')

    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file

    opt.lpcfg = check_file(opt.lpcfg)  # check file
    opt.lpnames = check_file(opt.lpnames)  # check file
    opt.virtualgate = check_file(opt.virtualgate)  # check file
    print(opt)

    with torch.no_grad():
        detect(opt.save_result)
