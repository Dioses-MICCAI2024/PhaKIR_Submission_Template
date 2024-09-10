import numpy as np
import csv
import json
import os
import os.path as osp
from tqdm import tqdm
import argparse 
import os
# from sample_videos import main

parser = argparse.ArgumentParser()
parser.add_argument('test_dir', type=str, help='Path of directory with the video directories for test')
parser.add_argument('out_dir', type=str, help='Path of directory with the video directories for test')
args = parser.parse_args()


# breakpoint()
# def format_videos(data_path):
data_path = args.test_dir
cont = 0
base_csv = []
# breakpoint()
for vid,video in tqdm(enumerate(sorted(os.listdir(osp.join(data_path))))):
    split = video.split('_')
    if osp.isdir(osp.join(data_path,video,'rgb')):
        # assert len(split)<=3 and len(split)>=2, 'Error in split {} {}'.format(len(split),video)
    
        video_frames = os.listdir(osp.join(data_path,video,'rgb'))
        video_frames.sort()

        for fid,frame in enumerate(video_frames):
            # if fid%10 ==0:
            cont+=1

            if int(frame.split('.')[0])%6 == 0:

                base_csv.append((video, vid+1, int(fid), osp.join(data_path, video,'rgb', '{}'.format(frame))))
# breakpoint()

os.makedirs(os.path.join(args.out_dir,'stuff_AR'),exist_ok=True)
with open(os.path.join(args.out_dir,'stuff_AR','inference.csv'), 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerows(base_csv)

