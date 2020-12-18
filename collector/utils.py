import cv2
import logging
import os
import glob
import tqdm
import numpy as np
import argparse
import shutil
logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('--input', type=str, required=False,
                        help='input video path')
    parser.add_argument('--out', type=str, required=False,
                        help='output images folder path')
    parser.add_argument('--skip', type=int, default=5,
                        help='number of frames to skip')
    return parser.parse_args()  

def vid_cap(out):
    video = cv2.VideoCapture(2)
    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4)) 
    size = (frame_width, frame_height) 
    cap = cv2.VideoWriter(out,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        30, size) 
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        cap.write(frame)
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("", frame)

        if key == ord("q"):
            break
    
    video.release() 
    cap.release() 
    cv2.destroyAllWindows()
    logging.info("video is saved at {}".format(out))
    

def extract_frames(args):
    os.makedirs(args.out, exist_ok=True)
    name_idx = np.max([i for i in range(len(args.input)) if args.input[i]=="/"])
    name = args.input[name_idx+1:]

    print(os.path.join(args.out, name, "invalid"))
    os.makedirs(os.path.join(args.out, name, "invalid"), exist_ok=True)
    os.makedirs(os.path.join(args.out, name, "invalid", "device"), exist_ok=True)
    os.makedirs(os.path.join(args.out, name, "invalid", "cover"), exist_ok=True)

    os.makedirs(os.path.join(args.out, name, "valid"), exist_ok=True)
    os.makedirs(os.path.join(args.out, name, "valid", "front"), exist_ok=True)
    os.makedirs(os.path.join(args.out, name, "valid", "side"), exist_ok=True)

    out_paths = [os.path.join(args.out, name, "invalid", "device"),
                 os.path.join(args.out, name, "invalid", "cover"),
                 os.path.join(args.out, name, "valid", "front"),
                 os.path.join(args.out, name, "valid", "side")]

    search_paths = [os.path.join(args.input, "invalid", "device", "*.mp4"),
                    os.path.join(args.input, "invalid", "cover", "*.mp4"),
                    os.path.join(args.input, "valid", "front", "*.mp4"),
                    os.path.join(args.input, "valid", "side", "*.mp4")]

    for i in range(4):
        try:
            for idx, video_path in tqdm.tqdm(enumerate(glob.glob(search_paths[i]))):
                cap = cv2.VideoCapture(video_path)
                flag = 0
                while(cap.isOpened()):
                    ret, frame = cap.read()

                    if ret==False:
                        break

                    if flag%5==0:
                        cv2.imwrite("{}/{}.{}.jpg".format(out_paths[i], idx, flag/5), frame)
                    flag += 1

                cap.release()
                cv2.destroyAllWindows()
        except:
            pass

def rotate():
    from PIL import Image
    for idx, img_path in tqdm.tqdm(enumerate(glob.glob("/home/pdd/Desktop/workspace/Valid_face_dataset/video_data_collector/output/macus/valid/front/*.jpg"))):
        img = Image.open(img_path)
        img = img.rotate(270)
        img.save('/home/pdd/Desktop/workspace/Valid_face_dataset/video_data_collector/output/macus/valid/new_front/{}.jpg'.format(idx))

def sort():
    os.makedirs("collection")
    os.makedirs(os.path.join("collection", "invalid"), exist_ok=True)
    os.makedirs(os.path.join("collection", "invalid", "device"), exist_ok=True)
    os.makedirs(os.path.join("collection", "invalid", "cover"), exist_ok=True)

    os.makedirs(os.path.join("collection", "valid"), exist_ok=True)
    os.makedirs(os.path.join("collection", "valid", "front"), exist_ok=True)
    os.makedirs(os.path.join("collection", "valid", "side"), exist_ok=True)

    for idx, folder_path in tqdm.tqdm(enumerate(glob.glob("output/*/*/*/*.jpg"))):
        if "device" in folder_path:
            shutil.copy(folder_path, "collection/invalid/device/{}.jpg".format(idx))
        if "cover" in folder_path:
            shutil.copy(folder_path, "collection/invalid/cover/{}.jpg".format(idx))
        if "front" in folder_path:
            shutil.copy(folder_path, "collection/valid/front/{}.jpg".format(idx))
        if "side" in folder_path:
            shutil.copy(folder_path, "collection/valid/side/{}.jpg".format(idx))

if __name__=="__main__":
    # args = parse_args()
    # # vid_cap("invalid_covered_face/covered_face.mp4")
    # for folder_path in glob.glob("input/*"):
    #     args.input = folder_path
    #     args.out = "output"
    #     extract_frames(args)
    vid_cap("testcam/test.mp4")