import glob
import tqdm
import os
import numpy as np
import argparse
import cv2
import shutil
import logging

logging.getLogger().setLevel(logging.INFO)

from retinaface.retina_detector import RetinaDetector
from PIL import Image

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='/home/pdd/Desktop/workspace/Valid_face_dataset/retinaface/weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--vis-thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--padding', type=int, default=5, help="padding to detect cropped face")
parser.add_argument('--invalid-out-path', default="/home/pdd/Desktop/workspace/Valid_face_dataset/new_dataset/fake", help="path to cropped face of real dtset")
args = parser.parse_args()

if __name__=="__main__":
    detector = RetinaDetector('cpu', args.trained_model, verbose=True)
    input_folder = "new_dataset/real/*.jpg"
    fake_idx = len(os.listdir("/home/pdd/Desktop/workspace/Valid_face_dataset/new_dataset/fake"))
    os.makedirs(args.invalid_out_path, exist_ok=True)

    for idx, img_path in tqdm.tqdm(enumerate(glob.glob(input_folder))):
        img = np.array(Image.open(img_path))
        dets = detector.detect_from_image(img)
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            b = list(map(int, b))
            h, w, _ = img.shape
            if b[0] - args.padding < 0 or b[2] + args.padding > h or b[1] - args.padding < 0 or b[3] + args.padding > w:
                shutil.move(img_path, os.path.join(args.invalid_out_path, "{}.jpg".format(fake_idx)))
                fake_idx+=1
                break

    # os.makedirs("./new_dataset/real", exist_ok=True)
    # os.makedirs("./new_dataset/fake", exist_ok=True)
    # [shutil.copy(img_path, "./new_dataset/real/{}.jpg".format(idx)) \
    #     for idx, img_path in tqdm.tqdm(enumerate(glob.glob("new_dataset/*/real*.jpg")))]
    # [shutil.copy(img_path, "./new_dataset/fake/{}.jpg".format(idx)) \
    #     for idx, img_path in tqdm.tqdm(enumerate(glob.glob("new_dataset/*/fake*.jpg")))]
    #fake 17670 real 21269 18967