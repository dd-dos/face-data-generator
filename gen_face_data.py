import glob
import tqdm
import os
import numpy as np
import cv2
import shutil
import random

from retinaface.retina_detector import RetinaDetector
from PIL import Image, ImageFilter
from utils import *

def gen_blur_face(args):
    '''
    blur image
    '''
    input_folder = os.path.join(args.input, "*.jpg")
    os.makedirs(args.output, exist_ok=True)

    for idx, img_path in tqdm.tqdm(enumerate(glob.glob(input_folder))):
        img = Image.open(img_path)
        blur_img = img.filter(ImageFilter.GaussianBlur(args.blur_radius))
        blur_img.save(os.path.join(args.output, "{}.jpg".format(idx)))

        if idx == args.nSample:
            return

def gen_black_covered_face(args):
    '''
    cover image with black region
    '''
    detector = RetinaDetector('cpu', "/home/pdd/Desktop/workspace/Valid_face_dataset/retinaface/weights/mobilenet0.25_Final.pth", verbose=True)
    input_folder = os.path.join(args.input, "*.jpg")
    idx = len(os.listdir('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid_black_cover'))
    for _, img_path in tqdm.tqdm(enumerate(glob.glob(input_folder))):
        img = Image.open(img_path)
        img  = cv2.imread(img_path)
        width, height, _ = img.shape
        dets = detector.detect_from_image(img)
        bound = [0,0,0,0,0]
        for det in dets:
            if det[4] >= args.vis_thres and get_area(det) >= get_area(bound):
                bound = list(map(int, det))

        if get_area(bound) == 0:
            continue

        box_height = bound[3] - bound[1]
        box_width = bound[2] - bound[0]
        
        img = Image.fromarray(img)
        if random.random() <= 0.25:
            img = img.crop((0,0,bound[2]-box_width/3,height))
        elif 0.25 < random.random() <= 0.5:
            img = img.crop((0,0,width,bound[3]-box_height/3))
        elif 0.5 < random.random() <= 0.75:
            img = img.crop((bound[0]+box_width/3, 0, width, height))
        elif 0.75 < random.random() <= 1:
            img = img.crop((0, bound[1]+box_height/3, width, height))

        try:
            img = np.array(img).astype(np.float32)
        except:
            import ipdb; ipdb.set_trace()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid_black_cover/{}.jpg".format(idx), img)
        idx += 1


def gen_fake_face(args):
    '''
    exchange faces between 2 images.
    '''
    detector = RetinaDetector('cpu', args.trained_model, verbose=True)
    input_folder = os.path.join(args.input, "*.jpg")
    os.makedirs(args.output, exist_ok=True)

    imgs_list_1 = list(glob.glob(input_folder))
    imgs_list_2 = imgs_list_1[1:-1] + [imgs_list_1[0]]

    for idx in tqdm.tqdm(range(len(imgs_list_1))):
        img_1 = np.array(Image.open(imgs_list_1[idx]))
        img_2 = np.array(Image.open(imgs_list_2[idx]))
        
        dets_1 = detector.detect_from_image(img_1)
        b1 = [0,0,0,0,0]
        for det in dets_1:
            if det[4] >= args.vis_thres and get_area(det) >= get_area(b1):
                b1 = list(map(int, det))

        if get_area(b1) == 0:
            continue

        dets_2 = detector.detect_from_image(img_2)
        b2 = [0,0,0,0,0]
        for det in dets_2:
            if det[4] >= args.vis_thres and get_area(det) >= get_area(b2):
                b2 = list(map(int, det))
                
        if get_area(b2) == 0:
            continue
        
        face_1 = img_1[b1[1]:b1[3], b1[0]:b1[2]]
        face_2 = img_2[b2[1]:b2[3], b2[0]:b2[2]]

        #size to resize to in (width, height) format
        size_1 = (b1[2]-b1[0], b1[3]-b1[1])
        size_2 = (b2[2]-b2[0], b2[3]-b2[1])
        
        face_1_new = cv2.resize(face_2, size_1, interpolation=cv2.INTER_CUBIC)
        face_2_new = cv2.resize(face_1, size_2, interpolation=cv2.INTER_CUBIC)
        
        img_1[b1[1]:b1[3], b1[0]:b1[2]] = face_1_new
        img_2[b2[1]:b2[3], b2[0]:b2[2]] = face_2_new

        Image.fromarray(img_1).save(os.path.join(args.output, "{}_1.jpg".format(idx)))
        Image.fromarray(img_2).save(os.path.join(args.output, "{}_2.jpg".format(idx)))
        
        if idx == args.nSample:
            return

def move():
    import shutil
    fake_idx = len(os.listdir('/home/pdd/Desktop/workspace/Valid_face_dataset/new_dataset/fake'))
    real_idx = len(os.listdir('/home/pdd/Desktop/workspace/Valid_face_dataset/new_dataset/real'))

    for idx, img_path in tqdm.tqdm(enumerate(glob.glob('/home/pdd/Desktop/workspace/Valid_face_dataset/video_data_collector/collection/*/*/*.jpg'))):
        if "device" in img_path:
            shutil.copy(img_path, "new_dataset/fake/{}.jpg".format(fake_idx))
            fake_idx += 1
        if "cover" in img_path:
            shutil.copy(img_path, "new_dataset/fake/{}.jpg".format(fake_idx))
            fake_idx += 1
        if "front" in img_path:
            shutil.copy(img_path, "new_dataset/real/{}.jpg".format(real_idx))
            real_idx += 1
        if "side" in img_path:
            shutil.copy(img_path, "new_dataset/real/{}.jpg".format(real_idx))
            real_idx += 1

if __name__ == "__main__":
    # valid_idx = 0
    # invalid_idx = 0
    # search_path = "/home/pdd/Desktop/workspace/Valid_face_dataset/valid_face_dataset/new_dataset/*/*.jpg"
    # for idx, img_path in tqdm.tqdm(enumerate(glob.glob(search_path))):
    #     if "fake" in img_path:
    #         shutil.copy(img_path, os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid','{}.jpg').format(invalid_idx))
    #         invalid_idx += 1
    #     elif "real" in img_path:
    #         shutil.copy(img_path, os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid','{}.jpg').format(valid_idx))
    #         valid_idx += 1

    # cover_idx = 0
    # device_idx = 0
    # search_path_2 = "/home/pdd/Desktop/workspace/Valid_face_dataset/collection/*/*/*.jpg"
    # for idx, img_path in tqdm.tqdm(enumerate(glob.glob(search_path_2))):
    #     if "cover" in img_path:
    #         shutil.copy(img_path, os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid_hand_cover','{}.jpg').format(cover_idx))
    #         cover_idx += 1
    #     if "device" in img_path:
    #         shutil.copy(img_path, os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid_device','{}.jpg').format(device_idx))
    #         device_idx += 1
    #     if "valid" in img_path:
    #         shutil.copy(img_path, os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid_device','{}.jpg').format(valid_idx))
    #         valid_idx += 1

    # search_path_2 = "/home/pdd/Desktop/workspace/Valid_face_dataset/collection/*/*/*.jpg"
    # detector = RetinaDetector('cpu', "/home/pdd/Desktop/workspace/Valid_face_dataset/retinaface/weights/mobilenet0.25_Final.pth")
    # cover_idx = 0
    # device_idx = 0
    # valid_idx = 0
    # invalid_idx = 0
    # for idx, img_path in tqdm.tqdm(enumerate(glob.glob(search_path_2))):
    #     img = Image.open(img_path)
    #     width, height = img.size
    #     dets = detector.detect_from_image(np.array(img))
    #     bound = [0,0,0,0,0]
    #     for det in dets:
    #         if det[4] >= 0.5 and get_area(det) >= get_area(bound):
    #             bound = list(map(int, det))

    #     if get_area(bound) == 0:
    #         continue
        
    #     x0 = max(bound[0]-((bound[3]-bound[1]))/2, 0)
    #     y0 = max(bound[1]-((bound[2]-bound[0]))/2, 0)
    #     x1 = min(bound[2]+((bound[3]-bound[1]))/2, width)
    #     y1 = min(bound[3]+((bound[2]-bound[0]))/2, height)

    #     img = img.crop((x0, y0, x1, y1))
    #     if "cover" in img_path:
    #         img.save(os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid_hand_cover','{}.jpg').format(cover_idx))
    #         cover_idx += 1
    #     if "device" in img_path:
    #         img.save(os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid_device','{}.jpg').format(device_idx))
    #         device_idx += 1
    #     if "front" in img_path or "side" in img_path:
    #         img.save(os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/valid','{}.jpg').format(valid_idx))
    #         valid_idx += 1

    # invalid_idx = len(os.listdir("/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid"))
    # valid_idx = len(os.listdir("/home/pdd/Desktop/workspace/Valid_face_dataset/final/valid"))
    # search_path = "/home/pdd/Desktop/workspace/Valid_face_dataset/valid_face_dataset/new_dataset/*/*.jpg"
    # for idx, img_path in tqdm.tqdm(enumerate(glob.glob(search_path))):
    #     if "fake" in img_path:
    #         shutil.copy(img_path, os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/invalid','{}.jpg').format(invalid_idx))
    #         invalid_idx += 1
    #     elif "real" in img_path:
    #         shutil.copy(img_path, os.path.join('/home/pdd/Desktop/workspace/Valid_face_dataset/final/valid','{}.jpg').format(valid_idx))
    #         valid_idx += 1
    args = parse_args()
    gen_black_covered_face(args)