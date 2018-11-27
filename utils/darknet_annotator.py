from ctypes import *
import math
import random
import os

DARKNET_BIN = "/home/dean/miniconda3/envs/ros-kache/lib/python3.6/site-packages/__libdarknet/libdarknet.so"

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



lib = CDLL(DARKNET_BIN, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def annotate(image_dir, cfg_path, weights_path, data_cfg_pth, detection_threshold = 0.25, img_width = 1280, img_height=1080):
    print("Initializing Annotation Pipeline via Darknet ... \o/ \o/")

    net = load_net(bytes(str(cfg_path), 'utf-8'), bytes(str(weights_path), 'utf-8') , 0)
    meta = load_meta(bytes(str(data_cfg_pth), 'utf-8'))
    all_anns = []


    if not os.path.isdir(image_dir):
        image_dirs = [image_dir]
    else: image_dirs = os.listdir(image_dir)

    for fname in image_dirs:
        anns = []
        if fname.endswith(".jpg"):

            r = detect(net, meta, bytes(str(os.path.join(image_dir, fname)), 'utf-8'))
            for detection in r:
                #print('DETECTION:', detection)
                d = {}
                d['category'] = detection[0].decode("utf-8")
                # Convert (x_center, y_center) to (x1, y1)
                new_y = float(detection[2][1]) - .5 * float(detection[2][3])
                new_x = float(detection[2][0]) - .5 * float(detection[2][2])

                corrected_detection = [ new_x, new_y, detection[2][2], detection[2][3]]
                d['box2d'] = {'x1': float(corrected_detection[0]),
                              'y1':float(corrected_detection[1]) ,
                              'x2': float(corrected_detection[0]) + float(corrected_detection[2]-1) ,
                              'y2': float(corrected_detection[1]) + float(corrected_detection[3])-1}
                d['manualShape'] = False

                if float(detection[1]) >= detection_threshold:
                    anns.append(d)
            all_anns.append((fname, anns))
    return all_anns

if __name__ == "__main__":
    image_dir = os.path.join('/media/dean/datastore1/datasets/kache_ai', 'frames_dev')
    cfg_path = "trainers/20181019--bdd-coco-ppl_1gpu_0001lr_256bat_32sd_90ep/cfg/yolov3-bdd100k.cfg",
    weights_path = "trainers/20181019--bdd-coco-ppl_1gpu_0001lr_256bat_32sd_90ep/backup/yolov3-bdd100k_final.weights",
    data_cfg_path = "trainers/20181019--bdd-coco-ppl_1gpu_0001lr_256bat_32sd_90ep/cfg/bdd100k.data"

    net = load_net(bytes(cfg_path, 'utf-8'), bytes(weights_path, 'utf-8') , 0)
    meta = load_meta(bytes(data_cfg_pth, 'utf-8'))
    for fname in os.listdir(image_dir):
        if fname.endswith(".jpg"):
            x = (os.path.join(directory, fname))
            r = detect(net, meta, bytes(os.path.join(directory, fname), 'utf-8'))
            #print('IMAGEPATH: {} |'.format(fname), r)
