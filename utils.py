import os
import numpy as np
import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='/home/pdd/Desktop/workspace/valid_face_dataset/retinaface/weights/mobilenet0.25_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--vis-thres', default=0.5, type=float, help='visualization_threshold')
    parser.add_argument('--input', type=str, required=True, help="input image folder")
    parser.add_argument('--output', type=str, required=True, help="output path")
    parser.add_argument('--nSample', type=int, default=1000, help="number of output samples")
    parser.add_argument('--padding-ratio', type=int, default=3, help="black padding ratio to valid image")
    parser.add_argument('--blur-radius', type=int, default=5, help='PIL image filter gaussian blur radius')
    return parser.parse_args()

def crop_hor(img, y, pos):
    '''
    img: cv2 image
    y: horizontal location to crop
    pos: top/bottom region to crop
    '''
    img = img.copy()
    if pos.lower() == 'top':
        img[:,:y] = 0
    elif pos.lower() == 'bot':
        img[:,y:] = 0
    else:
        raise Exception("Invalid region")
    return img

def crop_ver(img, x, pos):
    '''
    img: cv2 image
    x: vertical location to crop
    pos: left/right region  to crop
    '''
    img = img.copy()
    if pos.lower() == 'left':
        img[:x,:] = 0
    elif pos.lower() == 'right':
        img[x:,:] = 0
    else:
        raise Exception("Invalid region")
    return img

def get_area(det):
    return (det[3]-det[1]) * (det[2]-det[0])