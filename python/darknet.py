from ctypes import *
import random
import urllib.request
import cv2
import numpy as np
import threading

# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://192.168.1.75:8080/shot.jpg'

xThreshold = 100
yThreshold = 100

running = True

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

    

lib = CDLL("/home/kineret/code/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("libdarknet.so", RTLD_GLOBAL)
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

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

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

def detect1(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(bytes(image, encoding='utf-8'), 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)

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

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def objectInList(i, prevR):
    for k in prevR:
        if sameObject(i, k):
            return True
    return False

def sameObject(i, k):
    # object type
    if(i[0] != k[0]):
        return False

    if(abs(i[2][0] - k[2][0]) > xThreshold or abs(i[2][1] - k[2][1]) > yThreshold ):
        return False

    return True

def drawObject(i, img):
    print(i)
    x, y, w, h = i[2][0], \
                i[2][1], \
                i[2][2], \
                i[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(img, pt1, pt2, (13, 23, 227), 4)
    cv2.putText(img,
                i[0].decode() +
                " [" + str(round(i[1] * 100, 2)) + "]",
                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                [0, 255, 0], 2)

def detectLoop(net, meta):
    prevR = []
    while running:
        r = detect(net, meta)
        print(r)
        for i in r[0]:
            if(i[1] > 0.5):
                if not objectInList(i, prevR):
                    drawObject(i, r[1])
                
        prevR = r[0]
        if running:
            cv2.imshow("Detect", np.array(r[1], dtype = np.uint8 ))   
    print ("exiting from thread")

def detect(net, meta, thresh=.5, hier_thresh=.5, nms=.45):
    imgResp = urllib.request.urlopen(url)
    print("got here")
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp, cv2.IMREAD_UNCHANGED)

# https://github.com/pjreddie/darknet/issues/289#issuecomment-342448358
    data = img.ctypes.data_as(POINTER(c_ubyte))
    im = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): 
        do_nms_obj(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #cv2.imwrite('/home/kineret/code/try-cam/try.jpg', img)  
    free_image(im)
    free_detections(dets, num)
    return res, img     
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    net = load_net(bytes("/home/kineret/code/darknet/cfg/yolov3.cfg", encoding='utf-8'), bytes("/home/kineret/code/darknet/yolov3.weights", encoding='utf-8'),0)
    print("got here")
    meta = load_meta(bytes("/home/kineret/code/darknet/cfg/coco.data", encoding='utf-8'))

    cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)

    th = threading.Thread(target=detectLoop, args=(net, meta))
    th.start()

    cv2.waitKey()
    running = False

    print("Exiting")
    th.join()
