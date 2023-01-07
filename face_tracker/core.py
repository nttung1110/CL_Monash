# Extracting visual features representing for emotional signals of individual faces 
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import bbox_visualizer as bbv
import torch
import os.path as osp
import json


from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from dotmap import DotMap
from moviepy.editor import *
from tqdm import tqdm

def cal_iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

class VisualES():
    def __init__(self, args):
        self.args = args
    

    def initialize_model(self, face_detector, emotion_recognizer):
        self.face_detector = face_detector
        self.emotion_recognizer = emotion_recognizer

    
    def update_args(self, new_args):
        self.args = new_args
    

    def extract_single_frame(self, img):
        # detect bounding boxes
        bounding_boxes, probs = self.face_detector.detect(img, landmarks=False)
        if bounding_boxes is not None:
            bounding_boxes = bounding_boxes[probs>self.args.threshold_face] # threshold_face = 0.6


    def filter_noisy_track(self, all_tracks):
        # filter noisy track based on multiple criteria:
        #   1. Track length must exceed ...

        filter_tracks = []
        for each_track in all_tracks:
            length = len(each_track['frames_appear'])
            if length >= self.args.len_face_tracks:
                filter_tracks.append(each_track)

        return filter_tracks

    
    def interpolate_track(self, all_tracks):
        # interpolate gaps between frame with previous frame box
        interpolate_all_tracks = []

        for each_track in all_tracks:
            current_track = {"bbox": [], "id": each_track["id"], "frames_appear": []}

            start_frame = each_track["frames_appear"][0]
            end_frame = each_track["frames_appear"][1]

            run_frame = start_frame
            prev_frame = start_frame - 1
            prev_box = None

            for idx_box in range(len(each_track["bbox"])-1):
                frame_id = each_track["frames_appear"][idx_box]
                
                if frame_id - prev_frame > 1:
                    # should interpolate the middle, take the recent box to interpolate

                    times_repeat = frame_id - prev_frame - 1
                    all_middle_box = list(np.repeat([current_track["bbox"][-1]], times_repeat, 0))
                    current_track["bbox"].extend(all_middle_box)
                    current_track["frames_appear"].extend(range(prev_frame+1, frame_id))

                # add current
                current_track["bbox"].append(each_track["bbox"][idx_box])
                current_track["frames_appear"].append(frame_id)
                prev_frame = frame_id

            # remember to add the final frame
            current_track["bbox"].append(each_track["bbox"][-1])
            current_track["frames_appear"].append(each_track["frames_appear"][-1])

            interpolate_all_tracks.append(current_track)

        return interpolate_all_tracks


    def extract_sequence_frames(self, video_path):
        # finding all face tracks in video. A face track is defined as t = (l, t) where:
        #   + l represents for list of face location for that track
        #   + t represents for frame-index to the video of that track
        print("===========Finding face tracks==============")

        clip = VideoFileClip(video_path)
        frame_count = int(clip.fps * clip.duration)
        fps = int(clip.fps)
        width = int(clip.w)
        height = int(clip.h)
        frames_list = clip.iter_frames()
        self.args.skip_frame = int(fps/self.args.min_frame_per_second)

        softmax = torch.nn.Softmax(dim=1)
        
        all_tracks = []
        mark_old_track_idx = []
        idx_frame = -1

        all_emotion_category_tracks = []
        all_es_feat_tracks = []
        all_start_end_offset_track = []


        for idx, frame in tqdm(enumerate(frames_list)):

            # skip frame
            idx_frame += 1
            if idx_frame % self.args.skip_frame != 0:
                continue

            # FOR VISUALIZING ONLY
            draw_face_track_bbox = []
            progress_frame = str(idx_frame)+"/"+str(self.args.total_vid_frame)
                
            # print("Processing frame: ", progress_frame, video_path, self.args.batch_run, self.args.bin_run)

            # detect faces
            bounding_boxes, probs = self.face_detector.detect(frame, landmarks=False)
            if bounding_boxes is not None:
                bounding_boxes = bounding_boxes[probs>self.args.threshold_face] # threshold_face = 0.6

            if bounding_boxes is None:
                continue
            

            # Stage 1: Process dying tracks
            for idx, each_active_tracks in enumerate(all_tracks):
                old_idx_frame = each_active_tracks['frames_appear'][-1]

                if idx_frame - old_idx_frame > self.args.threshold_dying_track_len:
                    # this is the dying track, mark it
                    mark_old_track_idx.append(idx)

            # Stage 2: Assign new boxes to remaining active tracks or create a new track if there are active tracks

            for idx, bbox in enumerate(bounding_boxes):
                box = bbox.astype(int).tolist()
                # x1, y1, x2, y2 = box[0:4]   
                
                # check each track
                best_match_track_id = None
                best_match_track_score = 0


                # ====================
                # Stage 2.0: Extracting ES features from facial image
                [x1, y1, x2, y2] = box
                x1, x2  = min(max(0, x1), frame.shape[1]), min(max(0, x2), frame.shape[1])
                y1, y2 = min(max(0, y1), frame.shape[0]), min(max(0, y2), frame.shape[0])

                face_imgs = frame[y1:y2, x1:x2]
                
                # convert to rgb image
                face_imgs = face_imgs[:, :, ::-1]
                emotion, scores = self.emotion_recognizer.predict_emotions(face_imgs, logits=True)

                # softmax as feature
                scores = softmax(torch.Tensor(np.array([scores])))
                es_feature = scores[0].tolist() ### this is what we need
                emotion_cat = emotion ### this is what we need

                # raw feature
                # es_feature = scores.tolist()
                # emotion_cat = emotion

                # =====================
                # Stage 2.1: Finding to which track this es_feature belongs to based on iou
                for idx, each_active_tracks in enumerate(all_tracks):
                    if idx in mark_old_track_idx:
                        # ignore inactive track
                        continue
                    latest_track_box = each_active_tracks['bbox'][-1]

                    iou_score = cal_iou(latest_track_box, box)
                    
                    if iou_score > best_match_track_score and iou_score > self.args.threshold_iou_min_track:
                        best_match_track_id = idx
                        best_match_track_score = iou_score

                if best_match_track_id is None:
                    # there is no active track currently, then this will initialize a new track
                    new_track = {"bbox": [box], "id": len(all_tracks), 
                                "frames_appear": [idx_frame]}
                    all_tracks.append(new_track)

                    # also create new np array representing for new track here
                    new_es_array_track = np.array([es_feature])
                    all_es_feat_tracks.append(new_es_array_track)

                    # FOR VISUALIZING ONLY
                    if self.args.visualize_debug_face_track == True:
                        draw_face_track_bbox.append([box, new_track["id"]])
                else:
                    # update track
                    all_tracks[best_match_track_id]['bbox'].append(box)
                    all_tracks[best_match_track_id]['frames_appear'].append(idx_frame)

                    # update all_list

                    ### interpolate first

                    time_interpolate = idx_frame - all_tracks[best_match_track_id]['frames_appear'][-2] - 1

                    if time_interpolate > 0:
                        old_rep_track = all_es_feat_tracks[best_match_track_id][-1].tolist()
                        all_es_feat_tracks[best_match_track_id] = np.append(all_es_feat_tracks[best_match_track_id], [old_rep_track]*time_interpolate, axis=0)

                    ### then do update
                    all_es_feat_tracks[best_match_track_id] = np.append(all_es_feat_tracks[best_match_track_id], [es_feature], axis=0) # add more feature for this track

                    if self.args.visualize_debug_face_track == True:
                    # FOR VISUALIZING ONLY
                        draw_face_track_bbox.append([box, all_tracks[best_match_track_id]['id']])
                

            # FOR VISUALIZING ONLY, draw all face track box
            if self.args.visualize_debug_face_track == True:
                all_box = [l[0] for l in draw_face_track_bbox]
                all_id = ["ID:"+str(a[1]) for a in draw_face_track_bbox]

                # frame = bbv.draw_multiple_rectangles(frame, all_box)
                # frame = bbv.add_multiple_labels(frame, all_id, all_box)
                
                output.write(frame)

            if self.args.max_idx_frame_debug is not None:
                if idx_frame >= self.args.max_idx_frame_debug:
                    break

        if self.args.visualize_debug_face_track == True:
            output.release()

        # face track emotion results
        face_loc_results = {}
        face_emotional_feat_results = {}

        # all_tracks, all_es_feat_tracks, all_start_end_offset_track

        # filter again es signals, ignoring those tracks having length lesser than pre-defined numbers

        for idx, (es_feat_track, each_face_tracks) in enumerate(zip(all_es_feat_tracks, all_tracks)):
            length = es_feat_track.shape[0]
            if length >= self.args.len_face_tracks:
                # add to final result
                face_loc_results[idx] = each_face_tracks
                face_emotional_feat_results[idx] = es_feat_track.tolist()

        video_name = video_path.split('/')[-1].split('.')[0]

        write_json_loc_path = osp.join(self.args.path_face_result, 'loc_result', video_name+'.json')
        write_json_emot_feat_path = osp.join(self.args.path_face_result, 'emotion_feat_result', video_name+'.json')

        with open(write_json_loc_path, 'w') as fp:
            json.dump(face_loc_results, fp, indent=4)
        with open(write_json_emot_feat_path, 'w') as fp:
            json.dump(face_emotional_feat_results, fp, indent=4)
        
        return face_loc_results, face_emotional_feat_results

            
if __name__ == "__main__":
    # test args 
    args = DotMap()
    args.device = 'cuda'
    args.threshold_face = 0.6
    args.model_name = 'enet_b0_8_best_afew'
    args.threshold_dying_track_len = 30
    args.threshold_iou_min_track = 0.4
    # skip frame info
    args.min_frame_per_second = 3
    # debug mode
    args.max_idx_frame_debug = None
    args.len_face_tracks = 30

    # output json face
    args.path_face_result = '/home/nttung/research/Monash_CCU/CL_Monash/output_json/face_result'


    # test
    face_detector = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=args.device)
    emotion_recognizer = HSEmotionRecognizer(model_name=args.model_name, device=args.device)
    
    ES_extractor = VisualES(args)
    ES_extractor.initialize_model(face_detector, emotion_recognizer)
    path_test_video = '/home/nttung/research/Monash_CCU/mock_data/visual_data/format_mp4_video/M01000AJ6.mp4'

    
    a, b = ES_extractor.extract_sequence_frames(path_test_video)
    pdb.set_trace()

    # save feature for later usage
    # with open('test_es_feature.npy', 'wb') as f:
    #     np.save(f, np.array(all_es_feat_tracks))



