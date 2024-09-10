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
parser.add_argument('--test_dir', type=str, default='./inputs', help='Path of directory with the video directories for test')
parser.add_argument('--out_dir',type=str, default='./outputs', help='Path of directory with the video directories for test')
args = parser.parse_args()


# breakpoint()
# def format_videos(data_path):
data_path = args.test_dir
cont = 0
base_csv = []

base_categories = [
                            {
                                "id": 1,
                                "name": "Argonbeamer",
                                "description": "Argonbeamer",
                                "supercategory": "instrument",
                                "RGB-encoding": [60, 50, 50]
                            },
                            {
                                "id": 2,
                                "name": "Bipolar-Clamp",
                                "description": "Bipolar-Clamp",
                                "supercategory": "instrument",
                                "RGB-encoding": [89, 134, 179]
                            },
                            {
                                "id": 3,
                                "name": "Blunt-Grasper",
                                "description": "Blunt-Grasper",
                                "supercategory": "instrument",
                                "RGB-encoding": [128, 128, 128]
                            },
                            {
                                "id": 4,
                                "name": "Blunt-Grasper-Curved",
                                "description": "Blunt-Grasper-Curved",
                                "supercategory": "instrument",
                                "RGB-encoding": [200, 102, 235]
                            },
                            {
                                "id": 5,
                                "name": "Blunt-Grasper-Spec.",
                                "description": "Blunt-Grasper-Spec.",
                                "supercategory": "instrument",
                                "RGB-encoding": [179, 102, 235]
                            },
                            {
                                "id": 6,
                                "name": "Clip-Applicator",
                                "description": "Clip-Applicator",
                                "supercategory": "instrument",
                                "RGB-encoding": [0, 0, 255]
                            },
                            {
                                "id": 7,
                                "name": "Dissection-Hook",
                                "description": "Dissection-Hook",
                                "supercategory": "instrument",
                                "RGB-encoding": [80, 140, 0]
                            },
                            {
                                "id": 8,
                                "name": "Drainage",
                                "description": "Drainage",
                                "supercategory": "instrument",
                                "RGB-encoding": [255, 100, 0]
                            },
                            {
                                "id": 9,
                                "name": "Grasper",
                                "description": "Grasper",
                                "supercategory": "instrument",
                                "RGB-encoding": [255, 130, 0]
                            },
                            {
                                "id": 10,
                                "name": "HF-Coagulation-Probe",
                                "description": "HF-Coagulation-Probe",
                                "supercategory": "instrument",
                                "RGB-encoding": [255, 0, 153]
                            },
                            {
                                "id": 11,
                                "name": "Hook-Clamp",
                                "description": "Hook-Clamp",
                                "supercategory": "instrument",
                                "RGB-encoding": [0, 80, 80]
                            },
                            {
                                "id": 12,
                                "name": "Needle-Probe",
                                "description": "Needle-Probe",
                                "supercategory": "instrument",
                                "RGB-encoding": [204, 153, 153]
                            },
                            {
                                "id": 13,
                                "name": "Overholt",
                                "description": "Overholt",
                                "supercategory": "instrument",
                                "RGB-encoding": [255, 200, 170]
                            },
                            {
                                "id": 14,
                                "name": "Palpation-Probe",
                                "description": "Palpation-Probe",
                                "supercategory": "instrument",
                                "RGB-encoding": [255, 102, 255]
                            },
                            {
                                "id": 15,
                                "name": "PE-Forceps",
                                "description": "PE-Forceps",
                                "supercategory": "instrument",
                                "RGB-encoding": [30, 144, 1]
                            },
                            {
                                "id": 16,
                                "name": "Scissor",
                                "description": "Scissor",
                                "supercategory": "instrument",
                                "RGB-encoding": [255, 255, 0]
                            },
                            {
                                "id": 17,
                                "name": "Sponge-Clamp",
                                "description": "Sponge-Clamp",
                                "supercategory": "instrument",
                                "RGB-encoding": [40, 120, 80]
                            },
                            {
                                "id": 18,
                                "name": "Suction-Rod",
                                "description": "Suction-Rod",
                                "supercategory": "instrument",
                                "RGB-encoding": [153, 0, 204]
                            },
                            {
                                "id": 19,
                                "name": "Trocar-Tip",
                                "description": "Trocar-Tip",
                                "supercategory": "instrument",
                                "RGB-encoding": [153, 102, 0]
                            }
                        ]

base_categories_phases = [
                    {
                        "id": 0,
                        "name": "Preparation",
                        "description": "Preparation",
                        "supercategory": "phase"
                    },
                    {
                        "id": 1,
                        "name": "CalotTriangleDissection",
                        "description": "CalotTriangleDissection",
                        "supercategory": "phase"
                    },
                    {
                        "id": 2,
                        "name": "ClippingCutting",
                        "description": "ClippingCutting",
                        "supercategory": "phase"
                    },
                    {
                        "id": 3,
                        "name": "GallbladderDissection",
                        "description": "GallbladderDissection",
                        "supercategory": "phase"
                    },
                    {
                        "id": 4,
                        "name": "GallbladderRetraction",
                        "description": "GallbladderRetraction",
                        "supercategory": "phase"
                    },
                    {
                        "id": 5,
                        "name": "CleaningCoagulation",
                        "description": "CleaningCoagulation",
                        "supercategory": "phase"
                    },
                    {
                        "id": 6,
                        "name": "GallbladderPackaging",
                        "description": "GallbladderPackaging",
                        "supercategory": "phase"
                    }
                ]

base_images = []
base_annotations = []

img_id = 0
ann_id = 0

for vid,video in tqdm(enumerate(sorted(os.listdir(osp.join(data_path))))):
    split = video.split('_')
    if osp.isdir(osp.join(data_path,video,'Frames')):
        assert len(split)<=3 and len(split)>=2, 'Error kin split {} {}'.format(len(split),video)
    
        video_frames = os.listdir(osp.join(data_path,video,'Frames'))
        video_frames.sort()

        for fid,frame in enumerate(video_frames):
            # if fid%10 ==0:
            cont+=1
            #base_csv.append((video, vid+1, int(fid), osp.join(data_path, video,'Frames', '{}'.format(frame))))
            base_csv.append((video, vid+1, int(fid), osp.join(video,'Frames', '{}'.format(frame))))

            image_id = img_id
            base_images.append({
                "id": img_id,
                "file_name": osp.join(video,'Frames', '{}'.format(frame)),
                "width": 1920,  # Assuming fixed dimensions
                "height": 1080,  # Assuming fixed dimensions
                "date_captured": "",
                "license": 1,
                "coco_url": "",
                "flickr_url": "",
                "video_name": video,
                "frame_num": fid
            })

            # Invented polygon coordinates for segmentation
            segmentation = [
                [100, 200, 150, 250, 200, 200, 150, 150]
            ]
            bbox = [100, 150, 100, 100]  # Example bbox coordinates

            base_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "image_name": osp.join(video,'Frames', '{}'.format(frame)),
                "category_id": 1,  # Example category_id
                "iscrowd": 0,
                "area": 12740,  # Example area
                "segmentation": segmentation,
                "bbox": bbox,
                "actions": [0],
                "phases": 0,
                "steps": 0,
                "instruments": 1,
            })

            ann_id += 1
            img_id += 1
    
base_json = {
    "images": base_images,
    "annotations": base_annotations,
    "categories": base_categories,
    "phases_categories": base_categories_phases
}
    
    #print(len(annotations))
    #breakpoint()

os.makedirs(os.path.join(args.out_dir,'stuff_P', 'features'),exist_ok=True)

with open(os.path.join(args.out_dir, 'stuff_P', 'annotations.json'), 'w') as json_file:
    json.dump(base_json, json_file, indent=4)

with open(os.path.join(args.out_dir,'stuff_P','inference.csv'), 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerows(base_csv)

