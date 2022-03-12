import argparse
import cv2

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


###############################################################
class_index = { '0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,\
                'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18, 'J':19,\
                'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27, 'S':28, 'T':29,\
                'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35 }
###############################################
class Character():
    def __init__( self, Name, Location ):
        self.name = str(Name)
        self.location_X = int( ( Location[0] + Location[2] ) / 2 ) ### location[0] : x1，[2] : x2
        self.location_Y = int( ( Location[1] + Location[3] ) / 2 ) ### location[1] : y1，[3] : y2
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

###############################################
class LicensePlate():
    def __init__( self, Image, Center_Point, Bounding_Box, Vehicle_Class_Name ):
        self.image = Image
        self.centerPoint = Center_Point
        self.boundingBox = Bounding_Box
        self.vehicleClassName = Vehicle_Class_Name

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

###############################################
class Vehicle():
    def __init__( self, Class_Name, Bounding_Box ):
        self.className = Class_Name
        self.boundingBox = Bounding_Box

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

###############################################
def RectContains( rect, pt ):
    return rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]

###############################################
def FilterLicensePlateCandidate( candidate ):
    plate_list = []
    
    for i in range( len(candidate) ):
        plate_list.append(candidate[i][0])
        
    return plate_list

###############################################
def LicensePlateRule( LPlist, VehicleClassName ):
    length = len( LPlist )
    
    # sedan
    if VehicleClassName == 'sedan':
        # new sedan XXX-XXXX
        if length == 7:
            LPlist.insert( 3, '-' )
        elif length == 6:
            index = 0
            for i in range( len(LPlist) ):
                if class_index[ LPlist[ i ] ] > 9:
                    index = i
           
            if index > 3:  
                # old sedan XXXX-XX
                LPlist.insert( 4, '-' )
            elif index < 2:
                # old sedan XX-XXXX
                LPlist.insert( 2, '-' )
        else:
        # wrong character
            return ''
    # scooter            
    elif VehicleClassName == 'scooter':
        # new scooter XXX-XXXX
        # scooter XXX-XXX
        if length == 7 or length == 6:
            LPlist.insert( 3, '-' )
        else:
        # wrong character
            return ''
    # truck            
    elif VehicleClassName == 'truck':
        # new truck XXX-XXXX
        if length == 7:
            LPlist.insert( 3, '-' )
        elif length == 6:
            index = 0
            for i in range( len(LPlist) ):
                if class_index[ LPlist[ i ] ] > 9:
                    index = i
           
            if index > 3:  
                # old truck XXXX-XX
                LPlist.insert( 4, '-' )
            elif index < 2:
                # old truck XX-XXXX
                LPlist.insert( 2, '-' )
        elif length == 5:
            index = 0
            for i in range( len(LPlist) ):
                if class_index[ LPlist[ i ] ] > 9:
                    index = i
            
            if index > 2:  
                # truck XXX-XX
                LPlist.insert( 3, '-' )
            else:
                # old truck XX-XXX
                LPlist.insert( 2, '-' )
        elif length == 4:
            # truck XX-XX
            LPlist.insert( 2, '-' )
        else:
            # wrong character
            return ''
    # bus
    # trailer
    final = '%s'*len(LPlist) % tuple(LPlist)
    
    return final    
###############################################################

def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    lpweights = opt.lpweights

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

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


    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    lpmodel.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        lpmodel.fuse()
        lpimg = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        lpf = opt.lpweights.replace(opt.lpweights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(lpmodel, lpimg, lpf, verbose=False, opset_version=11,
                          input_names=['lpimages'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph

        lpmodel = onnx.load(lpf)  # Load the ONNX model
        onnx.checker.check_model(lpmodel)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(lpmodel.graph))  # Print a human readable representation of the graph

        return

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
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

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
    for path, img, im0s, vid_cap, cur_video_frame_cnt in dataset:
        print(path)
        need_reprocess = False
        LP_list = []
        Vehicle_list = []
        LP_xyxy_list = []

        dataset_count+=1
        #print(dataset_count)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        #t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

            no_license_detect = True

            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    # handle license plate class
                    if names[int(cls)] == 'license plate':
                        # crop license image
                        #print(int(xyxy[ 1 ]) , int(xyxy[ 3 ]), int(xyxy[ 0 ]) , int(xyxy[ 2 ]) )
                        lp_im0 = im0[ int(xyxy[ 1 ]) : int(xyxy[ 3 ]), int(xyxy[ 0 ]) : int(xyxy[ 2 ]) ]
                        # resize image height to 200
                        rate = 200 / ( int(xyxy[ 3 ]) - int(xyxy[ 1 ]) )
                        lp_im0 = cv2.resize(lp_im0, ( int( ( int(xyxy[ 2 ]) - int(xyxy[ 0 ]) ) * rate ), 200), interpolation=cv2.INTER_CUBIC)

                        # calculate LP image LP center point
                        centerX = int ( ( int( xyxy[ 0 ] ) + int( xyxy[ 2 ] ) ) / 2 )
                        centerY = int ( ( int( xyxy[ 1 ] ) + int( xyxy[ 3 ] ) ) / 2 )
                        
                        # create LicensePlate ( Image, Center_Point, Bounding_Box, Vehicle_Class_Name )
                        LP = LicensePlate( lp_im0, ( centerX, centerY ), 
                                        ( int( xyxy[ 0 ] ), int( xyxy[ 1 ] ), int( xyxy[ 2 ] ), int( xyxy[ 3 ] ) ), '' )

                        # add to LP_list
                        LP_list.append( LP )
                        LP_xyxy_list.append( xyxy )
                        need_reprocess = True
                        no_license_detect = False
                    # handle vehicle class
                    elif names[int(cls)] != 'license plate':
                        # create Vehicle( Center_Point, Bounding_Box, Vehicle_Class_Name )
                        vehicle = Vehicle( names[int(cls)], ( int( xyxy[ 0 ] ), int( xyxy[ 1 ] ), int( xyxy[ 2 ] ), int( xyxy[ 3 ] ) ) )
                        Vehicle_list.append( vehicle )

                        # draw result box
                        if save_img or view_img:  # Add bbox to image      
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box( xyxy, im0, label=label, color=colors[int(cls)] )

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    print('Stop Iteration!!')
                    raise StopIteration

            # Save results (image with detections)
            if save_img and no_license_detect:
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

        #print( "LP_list : ",len(LP_list) )
        #print( "LP_xyxy_list : ",len(LP_xyxy_list) )
        if need_reprocess:
            for i in range( len( LP_list ) ):
                final_LP = ''

                for j in range( len( Vehicle_list ) ):
                    # if LP center point in Vehicle bounding box:
                    if RectContains( Vehicle_list[ j ].boundingBox, LP_list[ i ].centerPoint ):
                        # LP class name = Vehicle class name
                        LP_list[ i ].vehicleClassName = Vehicle_list[ j ].className
                        break
                # detect character

                # padded resize
                lp_im0 = LP_list[ i ].image

                lp_img = letterbox( lp_im0, imgsz)[0]
                lp_img = lp_img[:, :, ::-1].transpose(2 , 0, 1)
                lp_img = np.ascontiguousarray(lp_img)

                lp_img = torch.from_numpy(lp_img).to(device)
                lp_img = lp_img.half() if half else lp_img.float()  # uint8 to fp16/32
                lp_img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if lp_img.ndimension() == 3:
                    lp_img = lp_img.unsqueeze(0)

                # Inference
                lp_pred = lpmodel( lp_img, augment=opt.augment )[ 0 ]
    
                # to float
                if half:
                    lp_pred = lp_pred.float()

                # Apply NMS
                lp_pred = non_max_suppression(lp_pred, opt.conf_thres, opt.iou_thres, multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

                for lp_i, lp_det in enumerate(lp_pred):  # detections for image lp_i
                    if lp_det is not None and len(lp_det):
                        # Rescale boxes from imgsz to im0 size
                        lp_det[:, :4] = scale_coords(lp_img.shape[2:], lp_det[:, :4], lp_im0.shape).round()

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

                        # 字元數量超過7個或低於4個即代表辨識有問題
                        if char_len > 7 or char_len < 4:
                            print("character length error!! len : ", char_len )
                            continue

                        # 用 x 座標由左到右排列
                        #print(license_plate_candidate)
                        license_plate_candidate = sorted(license_plate_candidate, key = lambda s: s[ 1 ])
                
                        #print(license_plate_candidate)
            
                        # 過濾車牌字元
                        license_plate = FilterLicensePlateCandidate( license_plate_candidate )
                        #print(license_plate)
            
                        final_LP = LicensePlateRule( license_plate, LP_list[ i ].vehicleClassName )

                if save_img or view_img:
                    plot_one_box( LP_xyxy_list[ i ], im0, label=final_LP, color=colors[ 0 ]) # 5 --> license plate color
            #####################################################################
            # Print time (inference + NMS)
	        #t2 = torch_utils.time_synchronized()
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

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
	#####################################################################
        # Print time (inference + NMS)
        t2 = torch_utils.time_synchronized()
        print('%sDone. (%.3fs)' % (s, t2 - t1))
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
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

    parser.add_argument('--lpcfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--lpnames', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--lpweights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')

    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file

    opt.lpcfg = check_file(opt.lpcfg)  # check file
    opt.lpnames = check_file(opt.lpnames)  # check file
    print(opt)

    with torch.no_grad():
        detect()
