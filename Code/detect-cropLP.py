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
    def __init__(self,name,location):
        self.name = str(name)
        self.location_X = int( ( location[0] + location[2] ) / 2 ) ### location[0] : x1，[2] : x2
        self.location_Y = int( ( location[1] + location[3] ) / 2 ) ### location[1] : y1，[3] : y2
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
###############################################
def FilterLicensePlateCandidate( candidate ):
    plate_list = []
    
    for i in range( len(candidate) ):
        plate_list.append(candidate[i][0])
        
    return plate_list
###############################################
def LicensePlateRule( LPlist ):
    length = len( LPlist )
    
    # new sedan/scooter XXX-XXXX
    if length == 7:
        LPlist.insert( 3, '-' )
    elif length == 6:
        index = 0
        for i in range( len(LPlist) ):
            if class_index[ LPlist[ i ] ] > 9:
                index = i
       
        if index > 2 and ( class_index[ LPlist[ 2 ] ] > 9 or class_index[ LPlist[ 3 ] ] > 9 ): 
            # 160-NLL
            # MXQ-162
            # scooter XXX-XXX
            LPlist.insert( 3, '-' )
        elif index > 3:  
            # old sedan XXXX-XX
            LPlist.insert( 4, '-' )
        elif index < 2:
            # old sedan XX-XXXX
            LPlist.insert( 2, '-' )
        else:           
            # scooter XXX-XXX
            LPlist.insert( 3, '-' )
    elif length == 5:
        index = 0
        for i in range( len(LPlist) ):
            if class_index[ LPlist[ i ] ] > 9:
                index = i
        
        if index > 2:  
            # truck XXX-XX
            LPlist.insert( 3, '-' )
        else:
            # old sedan XX-XXX
            LPlist.insert( 2, '-' )
    elif length == 4:
        # truck XX-XX
        LPlist.insert( 2, '-' )
    else:
        # wrong character
        return ''    
            
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
        dataset = LoadVideos(source, img_size=imgsz)

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

    LP_count = 0
    for path, img, im0s, vid_cap, cur_video_frame_cnt, cur_video_total_frame_cnt in dataset:
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
        t2 = torch_utils.time_synchronized()

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
                    
                    # not license plate class
                    if names[int(cls)] != 'license plate':
                        continue

            ###############################################################
                    # handle license plate class
                    if names[int(cls)] == 'license plate':
                        
                        LP_count += 1
                        LP_name = str(LP_count) + ".png"
                        save_path = str(Path(out) / LP_name)
                        print(save_path )
                        final_LP = ''
                        # crop license image
                        #print(int(xyxy[ 1 ]) , int(xyxy[ 3 ]), int(xyxy[ 0 ]) , int(xyxy[ 2 ]) )
                        lp_im0 = im0[ int(xyxy[ 1 ]) : int(xyxy[ 3 ]), int(xyxy[ 0 ]) : int(xyxy[ 2 ]) ]
                        # resize image height to 200
                        rate = 200 / ( int(xyxy[ 3 ]) - int(xyxy[ 1 ]) )
                        lp_im0 = cv2.resize(lp_im0, ( int( ( int(xyxy[ 2 ]) - int(xyxy[ 0 ]) ) * rate ), 200), interpolation=cv2.INTER_CUBIC)
                        
			# save license plate image
                        cv2.imwrite( save_path, lp_im0 )
                        continue
            
                    cls = cls1
                    if names[int(cls)] == 'license plate' and ( save_img or view_img ):
                        plot_one_box(xyxy, im0, label=final_LP, color=colors[int(cls)])
            
            ###############################################################
                    elif save_img or view_img:  # Add bbox to image      
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
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
                    continue
                    cv2.imwrite(save_path, im0)
                else:
                    continue
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
