import os
import pdb
import os.path as osp
import json
import librosa

from pyannote.audio import Pipeline
from tqdm import tqdm

if __name__ == "__main__":
    # define pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")

    path_audio_folder = '/home/nttung/research/Monash_CCU/mock_data/audio_data'
    path_out_json_save = '/home/nttung/research/Monash_CCU/CL_Monash/auxiliary_folder/SD_diar_result.json'

    path_cp_1 = '/home/nttung/research/Monash_CCU/mock_data/annotation/LDC2022E18_CCU_TA1_Mandarin_Chinese_Development_Annotation_V1.0/data/changepoint.tab'
    path_cp_2 = '/home/nttung/research/Monash_CCU/mock_data/annotation/LDC2022E18_CCU_TA1_Mandarin_Chinese_Development_Annotation_V2.0/data/changepoint.tab'

    # read cp
    with open(path_cp_1, 'r') as fp:
        cp1 = fp.read().split('\n')

    with open(path_cp_2, 'r') as fp:
        cp2 = fp.read().split('\n')

    cp_video_info = {}
    for each_line in cp1[1:-1]:
        each_line = each_line.split('\t')

        _, vid_name, cp_loc, _, _ = each_line
        if vid_name not in cp_video_info:
            cp_video_info[vid_name] = []

        if cp_loc not in cp_video_info[vid_name]:
            cp_video_info[vid_name].append(cp_loc)
    
    for each_line in cp2[1:-1]:
        each_line = each_line.split('\t')

        _, vid_name, cp_loc, _, _ = each_line
        if vid_name not in cp_video_info:
            cp_video_info[vid_name] = []

        if cp_loc not in cp_video_info[vid_name]:
            cp_video_info[vid_name].append(cp_loc)

    dict_result = {}

    # debuggg
    with open(path_out_json_save, 'r') as fp:
        dict_result = json.load(fp)

    for file in tqdm(os.listdir(path_audio_folder)):
        full_path_file = osp.join(path_audio_folder, file)
        duration_in_sec = librosa.get_duration(filename=full_path_file)
        file_name = file.split('.')[0]

        if file_name in dict_result:
            print("Exist", file_name)
            continue
        print("Processing", file)

        diarization = pipeline(full_path_file)

        D = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            D.append([int(turn.start), int(turn.end)])

        sorted_D = sorted(D, key=lambda x:x[1])
        dict_result[file_name] = {}

        if file_name in cp_video_info:
            # video has cp annotation, start segmenting

            # assuming the end point of video is also change point for quick computation
            cp_list = list(map(int, cp_video_info[file_name]))
            cp_list.append(int(duration_in_sec))

            current_cp_idx = 0
            current_SD_list = []

            dict_result[file_name]['SD_list'] = []
            
            for d_k in sorted_D:
                d_k_2 = d_k[1]
                if d_k_2 <= cp_list[current_cp_idx]:
                    current_SD_list.append(d_k)
                else:
                    # finish old SD, save and construct new ones
                    dict_result[file_name]['SD_list'].append(current_SD_list)

                    current_SD_list = []
                    current_cp_idx += 1

            # add to final SD
            if len(current_SD_list) > 0:
                dict_result[file_name]['SD_list'].append(current_SD_list)
        else:
            dict_result[file_name]['SD_list'] = [sorted_D] # only one segment for those non-cp annotation video

        dict_result[file_name]['all_D'] = sorted_D

    with open(path_out_json_save, 'w') as fp:
        json.dump(dict_result, fp, indent=4)



    
