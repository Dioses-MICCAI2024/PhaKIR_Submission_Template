#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from bdb import Breakpoint
from copy import deepcopy
import itertools
import os
import logging
import numpy as np
from tqdm import tqdm
import glob
import torch
from .psi_ava import Psi_ava_transformer

from copy import deepcopy
from .surgical_dataset import SurgicalDataset
from . import utils as utils

logger = logging.getLogger(__name__)

IDENT_FUNCT_DICT = {
                    'Phakirms': lambda x,y: 'Video_{:02d}/frame_{:06d}.png'.format(x,y),
                    }

class Phakirms(SurgicalDataset):
    """
    PSI-AVA dataloader.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "phakirms"
        self.zero_fill = 6
        self.image_type = "png"
        self.cfg = cfg
        self.multi_sample_rate = cfg.DATA.MULTI_SAMPLING_RATE
        super().__init__(cfg,split)
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        #breakpoint()
        return sec - 1
        
    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        # breakpoint()
        # Assuming self._image_paths is a list of lists of strings
        # for i, paths in enumerate(self._image_paths):
        #    self._image_paths[i] = [path.replace('\n', '') for path in paths]

        # Get the path of the middle frame 
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        #center_idx -= 1 # Because frames start in 1
        video_name = self._video_idx_to_name[video_idx]

        complete_name = 'frame_{}.{}'.format(str(sec).zfill(self.zero_fill), self.image_type)

        #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        path_complete_name = os.path.join(folder_to_images, 'Frames',complete_name)

        found_idx = self._image_paths[video_idx].index(path_complete_name)

        center_idx = found_idx

        # Get the frame idxs for current clip.
        sequence_pyramid = []

        for sample_rate in self.multi_sample_rate:
            seq_len = self._video_length * sample_rate
            seq = utils.get_sequence(
                center_idx,
                seq_len // 2,
                sample_rate,
                num_frames=len(self._image_paths[video_idx]),
                length = self._video_length,
                online = self.cfg.DATA.ONLINE,
            )

            sequence_pyramid.append(seq)

        assert center_idx in seq, f'Center index {center_idx} not in sequence {seq}'
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        
        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks}
        
        if self.cfg.FEATURES.ENABLE:
            rpn_features = []
            box_features = self.feature_boxes[complete_name] 

        if self.cfg.REGIONS.ENABLE:
            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                all_labels_presence = {f'{task}_presence':np.zeros(self._num_classes[task]) for task in self._region_tasks}
                all_labels.update(all_labels_presence)
            for box_labels in clip_label_list:
                if box_labels['bbox'] != [0,0,0,0]:
                    boxes.append(box_labels['bbox'])
                    if self.cfg.FEATURES.ENABLE:
                        rpn_box_key = tuple(box_labels['bbox'])
                        try:
                            features = np.array(box_features[rpn_box_key])
                            rpn_features.append(features)
                        except KeyError:
                            if self.cfg.ENDOVIS_DATASET.INCLUDE_GT and box_labels['is_gt']:
                                rpn_box_key = utils.get_best_features(box_labels["bbox"],box_features)
                                features = np.array(box_features[rpn_box_key])
                                rpn_features.append(features)
                            else:
                                raise ValueError(f"Predicted box {box_labels['bbox']} missing in features of {complete_name} {box_features.keys()}")

                    for task in self._region_tasks:
                        if isinstance(box_labels[task],list):
                            binary_task_label = np.zeros(self._num_classes[task],dtype='uint8')
                            box_task_labels = np.array(box_labels[task])-1
                            binary_task_label[box_task_labels] = 1
                            all_labels[task].append(binary_task_label)
                            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                                all_labels[f'{task}_presence'][box_task_labels] = 1
                        elif isinstance(box_labels[task],int):
                            all_labels[task].append(box_labels[task]-1)
                            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                                all_labels[f'{task}_presence'][box_labels[task]-1] = 1
                        else:
                            raise ValueError(f'Do not support annotation {box_labels[task]} of type {type(box_labels[task])} in frame {complete_name}')
        else:
            for task in self._region_tasks:
                binary_task_label = np.zeros(self._num_classes[task]+1, dtype='uint8')
                label_list = [label[task] for label in clip_label_list]
                assert all(type(label_list[0])==type(lab_item) for lab_item in label_list), f'Inconsistent label type {label_list} in frame {complete_name}'
                if isinstance(label_list[0], list):
                    label_list = set(list(itertools.chain(*label_list)))
                    binary_task_label[label_list] = 1
                elif isinstance(label_list[0], int):
                    label_list = set(label_list)
                    binary_task_label[label_list] = 1
                else:
                    raise ValueError(f'Do not support annotation {label_list[0]} of type {type(label_list[0])} in frame {complete_name}')
                all_labels[task] = binary_task_label[1:]

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}
        if self.cfg.REGIONS.ENABLE:
            max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self.cfg.ENDOVIS_DATASET.INCLUDE_GT else self.cfg.DATA.MAX_BBOXES
            if  len(boxes):
                ori_boxes = deepcopy(boxes)
                boxes = np.array(boxes)
                if self.cfg.FEATURES.ENABLE:
                    rpn_features = np.array(rpn_features)
            else:
                ori_boxes = []
                boxes = np.zeros((max_boxes, 4))
        else:
            boxes = np.zeros((1, 4))
                
        # Load images of current clip.
        #breakpoint()
        images_pyramid = []
        for sequence in sequence_pyramid:
            image_paths = [self._image_paths[video_idx][frame] for frame in sequence]
            imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND
            )
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                imgs, boxes=boxes
            )
            imgs = utils.pack_pathway_output(self.cfg, imgs)
            images_pyramid.append(imgs)
        
        # Padding and masking for a consistent dimensions in batch
        if self.cfg.REGIONS.ENABLE and len(ori_boxes):
            max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self.cfg.ENDOVIS_DATASET.INCLUDE_GT else self.cfg.DATA.MAX_BBOXES

            # TODO: REMOVE when all done
            assert len(boxes)==len(ori_boxes)==len(rpn_features), f'Inconsistent lengths {len(boxes)} {len(ori_boxes)} {len(rpn_features)}'
            assert len(boxes)<= max_boxes and len(ori_boxes)<=max_boxes and len(rpn_features)<=max_boxes, f'More boxes than max box num {len(boxes)} {len(ori_boxes)} {len(rpn_features)}'

            bbox_mask = np.zeros(max_boxes,dtype=bool)
            bbox_mask[:len(boxes)] = True
            extra_data["boxes_mask"] = bbox_mask

            if len(boxes)<max_boxes:
                c_boxes = np.concatenate((boxes,np.zeros((max_boxes-len(boxes),4))),axis=0)
                boxes = c_boxes
            extra_data["ori_boxes"] = ori_boxes
            extra_data["boxes"] = boxes

            if self.cfg.FEATURES.ENABLE:
                if len(rpn_features)<max_boxes:
                    c_rpn_features = np.concatenate((rpn_features,np.zeros((max_boxes-len(rpn_features), 3796 if self.cfg.FEATURES.MODEL=='detr' else (512 if self.cfg.FEATURES.MODEL=='m2f' else 1024)))),axis=0)
                    rpn_features = c_rpn_features
                extra_data["rpn_features"] = rpn_features
        elif self.cfg.REGIONS.ENABLE:
            bbox_mask = np.zeros(max_boxes,dtype=bool)
            extra_data["boxes_mask"] = bbox_mask
            extra_data["ori_boxes"] = ori_boxes
            extra_data["boxes"] = boxes

            if self.cfg.FEATURES.ENABLE:
                extra_data["rpn_features"] = np.zeros((max_boxes, self.cfg.FEATURES.DIM_FEATURES))

        if self.cfg.NUM_GPUS>1:
            video_num = int(video_name.replace('Video_',''))
            frame_identifier = [video_num,sec]
        else:
            frame_identifier = f'{video_name}/{complete_name}'
        
        return images_pyramid, all_labels, extra_data, frame_identifier


class Phakir_transformer(Psi_ava_transformer):
    """
    PSI-AVA dataloader.
    """
    def __init__(self, cfg, split, include_subvideo=False):
        super().__init__(cfg,split)
        self.include_subvideo = include_subvideo
        self.fps = 1
        self.image_type = "png"

        self.dataset_name = "phakir"
        self.zero_fill = 6
        self.cfg = cfg

    def keyframe_mapping(self, video_idx, sec_idx, sec):
        return sec - 1

    def _load_samples_features_old(self, samples, cfg, include_subvideo=False):
        if self._split == "train":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_TRAIN
        elif self._split == "val":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_VAL

        output_features = []
        
        for img in samples:
            #img = inverted_assignation[img]
            if include_subvideo:
                case, subvideo, frame = img.split("/")[-3:]

                frame = f"{frame[:-4]}.pth"
                
                feature_path = os.path.join(feat_path, case, subvideo, frame)
                feature_list = torch.load(feature_path)

                try:
                    if isinstance(feature_list, torch.Tensor):
                        feat = np.array(feature_list)
                    else:
                        feat = np.concatenate(feature_list)
                except:
                    breakpoint()
            else:
                case, frame = img.split("/")[-2:]

                frame = f"{frame[:-4]}.pth"
                
                feature_path = os.path.join(feat_path, case, frame)
                feature_list = torch.load(feature_path)

                try:
                    if isinstance(feature_list, torch.Tensor):
                        feat = np.array(feature_list)
                    else:
                        feat = np.concatenate(feature_list)
                except:
                    breakpoint()

            output_features.append(feat.tolist())

        return output_features
    
    def _load_features(self, cfg):
        if self._split == "train":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_TRAIN
        elif self._split == "val":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_VAL

        videos_list = os.listdir(feat_path)
        feats_dict = {}

        for video in tqdm(videos_list, desc="Getting video features..."):
            feat_paths = glob.glob(os.path.join(feat_path,video, '*.pth'))
            for feature in feat_paths:
                feat_list = torch.load(feature)
                try:
                    if isinstance(feat_list, torch.Tensor):
                        feat = np.array(feat_list)
                    else:
                        feat = np.concatenate(feat_list)
                except:
                    breakpoint()

                case, img_name = feature.split("/")[-2:]
                img_key = f"{case}/{img_name[:-4]}.png"
                feats_dict[img_key] = feat
        
        self._feats_dict = feats_dict

    def _load_samples_features(self, samples, cfg, include_subvideo=False):
        if self._split == "train":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_TRAIN
        elif self._split == "val":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_VAL

        output_features = []

        for img in samples:
            if include_subvideo:
                case, subvideo, frame = img.split("/")[-2:]
            else:
                case, frame = img.split("/")[-2:]
            output_features.append(self._feats_dict[img]) 

        return output_features
    
    def _get_feature_path_names(self, image_paths):
        
        features_paths = []
        for original_path in image_paths:
            path_parts = original_path.split('/')
            case_folder = path_parts[-2]
            image_filename = path_parts[-1]
            new_path = os.path.join(case_folder, image_filename)
            features_paths.append(new_path)

        return features_paths
    
    def _fill_feats_with_zeros(self,assigned_boxes_tube, fixed_length):

        for key in assigned_boxes_tube:
            while len(assigned_boxes_tube[key]) < fixed_length:
                assigned_boxes_tube[key].append((0.0, 0.0, 0.0, 0.0))
            assert len(assigned_boxes_tube[key]) == fixed_length, 'Not the desired length'
        
        return assigned_boxes_tube

    def _generate_mask_tube(self, box_features_clip, center_idx_feats, feature_names, rpn_box_key):
        center_features = box_features_clip[center_idx_feats]
        past_features = box_features_clip[:center_idx_feats]
        # Get past features in order closest to central frame -> farthest to central frame
        past_features.reverse()
        future_features = box_features_clip[center_idx_feats + 1:]
        
        actual_feats = center_features
        assigned_boxes_tube_past = {i: [] for i in center_features.keys()}
        assigned_boxes_tube_future = {i: [] for i in center_features.keys()}
        
        past_len = len(past_features)
        future_len = len(future_features)
        
        # Process future features
        for feat in future_features:
            next_feats = feat
            assigned_boxes_tube_future, next_feats = self.process_features(actual_feats, next_feats, 
                                                            assigned_boxes_tube_future)
             
            assigned_boxes = list(assigned_boxes_tube_future.values())
            assigned_boxes = [item for row in assigned_boxes for item in row]
            # Only include the copy of the assigned feats from the previous iteration
            actual_feats = feat.copy()
        
        
        actual_feats = center_features
        # Process past features
        for feat in past_features:
            prev_feats = feat
            
            assigned_boxes_tube_past, prev_feats = self.process_features(actual_feats, prev_feats, 
                                                             assigned_boxes_tube_past)
            
            assigned_boxes = list(assigned_boxes_tube_past.values())
            assigned_boxes = [item for row in assigned_boxes for item in row]
            actual_feats = feat.copy()

        assigned_boxes_tube_past = self._fill_feats_with_zeros(assigned_boxes_tube_past, past_len)
        assigned_boxes_tube_future = self._fill_feats_with_zeros(assigned_boxes_tube_future, future_len)

        # Reverse lists in past tube
        assigned_boxes_tube_past = {key: [values[i] for i in range(len(values)-1, -1, -1)] 
                                    for key, values in assigned_boxes_tube_past.items()}

        assert assigned_boxes_tube_past.keys() == assigned_boxes_tube_future.keys(), "Keys are not the same"
        
        boxes_keys = list(assigned_boxes_tube_past.keys())

        # Join past, present, and future tube
        features_clip_bboxes = {i: assigned_boxes_tube_past[i] +  [i] + 
                         assigned_boxes_tube_future[i] for i in boxes_keys}

        # Create feature cube
        features = np.zeros((self._video_length, self.ins_feature_dim))

        seq_bbox_mask = np.zeros((self._video_length), dtype=bool)  # Initialize a mask

        rpn_box_key = max(features_clip_bboxes.keys(), key=lambda key: self.calculate_iou_2(key, rpn_box_key))
        for central_frame_idx, central_box in enumerate(features_clip_bboxes[rpn_box_key]):
            complete_tube_features = []
            for idx, bbox in enumerate(features_clip_bboxes[rpn_box_key]):
                if tuple(bbox) in box_features_clip[idx]:
                    assigned_frame_feature = np.array(box_features_clip[idx][tuple(bbox)])
                else:
                    assigned_frame_feature = np.zeros((self.ins_feature_dim))
                    seq_bbox_mask[idx] = True
                complete_tube_features.append(assigned_frame_feature)

            features[:, :] = complete_tube_features
            #features[central_frame_idx,:, :] = complete_tube_features

        #features = features.reshape((features.shape[0], -1))
        features = features.reshape((-1))
        
        return features, seq_bbox_mask
    
    def process_features(self, actual_feats, next_feats, assigned_boxes_tube):
        actual_bboxes = list(actual_feats.keys())
        next_bboxes = list(next_feats.keys())

        actual_bboxes = [list(i) for i in actual_bboxes]
        next_bboxes = [list(i) for i in next_bboxes]

        assigned_boxes = self._match_boxes(actual_bboxes, next_bboxes)
        #if '0.0 0.0 0.0 0.0' in assigned_boxes:
            
        for key_tube, value_tube in assigned_boxes_tube.items():
            if key_tube in assigned_boxes:
                assigned_boxes_tube[key_tube].append(tuple(assigned_boxes[key_tube]))
            else:
                for ass_key, ass_val in assigned_boxes.items():
                    if ass_key in value_tube:
                        assigned_boxes_tube[key_tube].append(tuple(ass_val))
                        #if ass_val == '0.0 0.0 0.0 0.0':
                        #    next_feats['0.0 0.0 0.0 0.0'] = [0] * 256

        return assigned_boxes_tube, next_feats

    def transform_bboxes(self, bboxes, scale='down'):
        for idx, bbox in enumerate(bboxes):
            bbox = [float(i) for i in bbox]
            width = 1280
            height = 800
            if scale =='down':
                bbox[0] /= width 
                bbox[2] /= width
                bbox[1] /= height
                bbox[3] /= height
                bboxes[idx] = bbox
            elif scale=='up':
                bbox[0] *= width 
                bbox[2] *= width
                bbox[1] *= height
                bbox[3] *= height
                bboxes[idx] = bbox

        return bboxes

    def _match_boxes(self, actual_bboxes, next_bboxes):
        similarity_matrix = np.zeros((len(actual_bboxes), len(next_bboxes)))
        for i, ann1 in enumerate(actual_bboxes):
            for j, ann2 in enumerate(next_bboxes):
                iou =  self._calculate_iou(ann1, ann2)
                similarity_matrix[i, j] = iou

        actual_bboxes_indices, next_bboxes_indices = linear_sum_assignment(-similarity_matrix)
        assigned_boxes = {}

        for actual_index, next_index in zip(actual_bboxes_indices, next_bboxes_indices):
            actual_box = actual_bboxes[actual_index]

            next_box = next_bboxes[next_index]

            assigned_boxes[tuple(actual_box)] = next_box

        if len(assigned_boxes) != len(actual_bboxes):
            bboxes = list(assigned_boxes.keys())
            missing_boxes = [box for box in actual_bboxes if box not in bboxes]
            for box in missing_boxes:
                assigned_boxes[tuple(box)] = (0.0, 0.0, 0.0, 0.0)
            
            #print("No instrument found in the next one")

        return assigned_boxes

    def _calculate_iou(self, boxA, boxB):
        # boxA: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

        return iou

    def calculate_iou_2(self, box1, box2):
        """
        Calculate the intersection over union (IoU) metric between two bounding boxes.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate coordinates of intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        # Calculate area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        return iou

    def get_bbox_feats(self, box_labels, box_features, complete_name):
        rpn_box_key = tuple(box_labels['bbox'])
        try:
            rpn_box_key = max(box_features.keys(), key=lambda key: self.calculate_iou_2(key, rpn_box_key))
            features = np.array(box_features[rpn_box_key])
        except KeyError:
            if self.cfg.ENDOVIS_DATASET.INCLUDE_GT and box_labels['is_gt']:
                rpn_box_key = utils.get_best_features(box_labels["bbox"],box_features)
                features = np.array(box_features[rpn_box_key])
            else:
                raise ValueError(f"Predicted box {box_labels['bbox']} missing in features of {complete_name} {box_features.keys()}")
        return features

    def get_feature_paths_per_case(self,feature_paths, include_subvideo):
        case_dict = {}
        for path in feature_paths:
            parts = path.split('/')
            case_number = parts[0] 
            if case_number in case_dict:
                case_dict[case_number].append(path)
            else:
                case_dict[case_number] = [path]
        return case_dict
    
    def get_temporal_feature_paths_per_case(self, feature_paths, include_subvideo=False):
        case_dict = {}

        if not include_subvideo:
            for case in os.listdir(feature_paths):
                if case not in case_dict:
                    case_dict[case] = []
                case_path = os.path.join(feature_paths, case)
            
                for frame in os.listdir(case_path):
                    frame_path = os.path.join(case, frame).replace('pth', "png")
                    if self.do_assignation:
                        frame_path = self.assignation[frame_path]
                    case_dict[case].append(frame_path)
        else:
            for case in os.listdir(feature_paths):
                if case not in case_dict:
                    case_dict[case] = []
                case_path = os.path.join(feature_paths, case)

                for subvideo in os.listdir(case_path):
                    subvideo_path = os.path.join(case_path, subvideo)
                    
                    for frame in os.listdir(subvideo_path):
                        frame_path = os.path.join(case, subvideo, frame).replace('pth',"png")
                        if self.do_assignation:
                            frame_path = self.assignation[frame_path]
                        case_dict[case].append(frame_path)

        return case_dict

    def __getitem__(self, idx):

        # Get the path of the middle frame 
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        complete_name = 'frame_{}.{}'.format(str(sec).zfill(self.zero_fill), self.image_type)

         #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        path_complete_name = os.path.join(folder_to_images, "Frames", complete_name)
        
        found_idx = self._image_paths[video_idx].index(path_complete_name)
        
        #self.feature_paths[video_name].sort()
        # Sort the list based on the numeric value in the frame file name
        self.feature_paths[video_name] = sorted(self.feature_paths[video_name], key=lambda x: int(x.split('_')[-1].split('.')[0]))

        video_feat_paths = self.feature_paths[video_name]
        
        #feat_idx = video_feat_paths.index(complete_name)
        # subvideo_complete_name = os.path.join(video_name, complete_name)
        feat_idx = video_feat_paths.index(os.path.join(video_name, complete_name))

        seq_len = self._video_length * self._sample_rate

        seq_feats = utils.get_sequence(
                    feat_idx,
                    seq_len // 2,
                    self._sample_rate,
                    num_frames=len(video_feat_paths),
                    length = seq_len,
                    online = self.cfg.DATA.ONLINE,
                )

        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])

        assert len(clip_label_list) > 0

        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks} 


        feature_names = [video_feat_paths[frame] for frame in seq_feats]

        # Get boxes and labels for current clip.
        boxes = []
        
        if self.cfg.FEATURES.ENABLE:
            rpn_features = []
            seq_mask = []
            box_features = self.feature_boxes[complete_name] 

        if self.cfg.REGIONS.ENABLE:
            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                all_labels_presence = {f'{task}_presence':np.zeros(self._num_classes[task]) for task in self._region_tasks}
                all_labels.update(all_labels_presence)
            for box_labels in clip_label_list:
                if box_labels['bbox'] != [0,0,0,0]:
                    boxes.append(box_labels['bbox'])
                    if self.cfg.FEATURES.ENABLE:
                        features = self.get_bbox_feats(box_labels, box_features, complete_name)

                        rpn_box_key = tuple(box_labels['bbox'])
                        
                        boxes_central_frame = list(box_features.keys())
                        
                        box_features_clip = [self.feature_boxes[i] for i in feature_names]
                        
                        center_idx_feats = seq_feats.index(feat_idx)

                        features, seq_bbox_mask = self._generate_mask_tube(box_features_clip, center_idx_feats, feature_names, rpn_box_key)
                        
                        rpn_features.append(features)
                        seq_mask.append(seq_bbox_mask)


                    for task in self._region_tasks:
                        if isinstance(box_labels[task],list):
                            binary_task_label = np.zeros(self._num_classes[task],dtype='uint8')
                            box_task_labels = np.array(box_labels[task])-1
                            binary_task_label[box_task_labels] = 1
                            all_labels[task].append(binary_task_label)
                            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                                all_labels[f'{task}_presence'][box_task_labels] = 1
                        elif isinstance(box_labels[task],int):
                            all_labels[task].append(box_labels[task]-1)
                            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                                all_labels[f'{task}_presence'][box_labels[task]-1] = 1
                        else:
                            raise ValueError(f'Do not support annotation {box_labels[task]} of type {type(box_labels[task])} in frame {complete_name}')
        else:
            for task in self._region_tasks:
                binary_task_label = np.zeros(self._num_classes[task]+1, dtype='uint8')
                label_list = [label[task] for label in clip_label_list]
                assert all(type(label_list[0])==type(lab_item) for lab_item in label_list), f'Inconsistent label type {label_list} in frame {complete_name}'
                if isinstance(label_list[0], list):
                    label_list = set(list(itertools.chain(*label_list)))
                    binary_task_label[label_list] = 1
                elif isinstance(label_list[0], int):
                    label_list = set(label_list)
                    binary_task_label[label_list] = 1
                else:
                    raise ValueError(f'Do not support annotation {label_list[0]} of type {type(label_list[0])} in frame {complete_name}')
                all_labels[task] = binary_task_label[1:]

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}
        if self.cfg.REGIONS.ENABLE:
            max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self.cfg.ENDOVIS_DATASET.INCLUDE_GT else self.cfg.DATA.MAX_BBOXES
            if  len(boxes):
                ori_boxes = deepcopy(boxes)
                boxes = np.array(boxes)
                if self.cfg.FEATURES.ENABLE:
                    rpn_features = np.array(rpn_features)
            else:
                ori_boxes = []
                boxes = np.zeros((max_boxes, 4))
        else:
            boxes = np.zeros((1, 4))
                

        # Padding and masking for a consistent dimensions in batch
        if self.cfg.REGIONS.ENABLE and len(ori_boxes):
            max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self.cfg.ENDOVIS_DATASET.INCLUDE_GT else self.cfg.DATA.MAX_BBOXES

            bbox_mask = np.zeros(max_boxes,dtype=bool)
            bbox_mask[:len(boxes)] = True
            extra_data["boxes_mask"] = bbox_mask

            if len(boxes)<max_boxes:
                c_boxes = np.concatenate((boxes,np.zeros((max_boxes-len(boxes),4))),axis=0)
                boxes = c_boxes
            extra_data["ori_boxes"] = ori_boxes
            extra_data["boxes"] = boxes

            if self.cfg.FEATURES.ENABLE:
                if len(rpn_features)<max_boxes:
                    c_rpn_features = np.concatenate((rpn_features,np.zeros((max_boxes-len(rpn_features), (self.ins_feature_dim * self._seq_len) if self.cfg.FEATURES.MODEL=='detr' else (512 if self.cfg.FEATURES.MODEL=='m2f' else 1024)))),axis=0)
                    rpn_features = c_rpn_features

                    c_seq_mask = np.concatenate((seq_mask,np.zeros((max_boxes - len(seq_mask), self._video_length), dtype=bool)),axis=0)
                    seq_mask = c_seq_mask 

                seq_mask = np.array(seq_mask).reshape(-1)
                extra_data["sequence_mask"] = seq_mask
                extra_data["rpn_features"] = torch.Tensor(rpn_features)

        elif self.cfg.REGIONS.ENABLE:
            bbox_mask = np.zeros(max_boxes,dtype=bool)
            extra_data["boxes_mask"] = bbox_mask
            extra_data["ori_boxes"] = ori_boxes
            extra_data["boxes"] = boxes

            if self.cfg.FEATURES.ENABLE:
                extra_data["rpn_features"] = np.zeros((max_boxes, self.cfg.FEATURES.DIM_FEATURES))
        
        seq_mask = torch.tensor([False])
        extra_data["sequence_mask"] = seq_mask

        seq_feat = feature_names

        #load features 
        features = self._load_samples_features_old(seq_feat, self.cfg)
        temporal_features = torch.tensor(features)

        if self.cfg.NUM_GPUS>1:
            video_num = int(video_name.replace('Video_',''))
            frame_identifier = [video_num,sec]
        else:
            frame_identifier =f'{video_name}/{complete_name}' 

        return temporal_features, all_labels, extra_data, frame_identifier