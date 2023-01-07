from core import *

if __name__ == "__main__":
    # test args 
    args = DotMap()
    args.device = 'cuda:0'
    args.threshold_face = 0.6
    args.model_name = 'enet_b0_8_best_afew'
    args.threshold_dying_track_len = 30
    args.threshold_iou_min_track = 0.4
    # skip frame info
    args.min_frame_per_second = 3
    # debug mode
    args.max_idx_frame_debug = None
    args.len_face_tracks = 30
    args.path_face_result = '/home/nttung/research/Monash_CCU/CL_Monash/output_json/face_result'

    path_all_video = '/home/nttung/research/Monash_CCU/mock_data/visual_data/format_mp4_video'

    # initialize all models
    face_detector = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=args.device)
    emotion_recognizer = HSEmotionRecognizer(model_name=args.model_name, device=args.device)
    
    ES_extractor = VisualES(args)
    ES_extractor.initialize_model(face_detector, emotion_recognizer)

    for file in tqdm(os.listdir(path_all_video)):
        print(file)
        path_video = osp.join(path_all_video, file)
        
        a, b = ES_extractor.extract_sequence_frames(path_video)
        
