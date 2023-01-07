import os
import os.path as osp
import pdb

path_video_folder_ldcc_in = '/home/nttung/research/Monash_CCU/mock_data/all_vid_data/LDC2022E19_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R2_V1.0/tools/dl_tool/out'
path_video_out_folder_mp4 = '/home/nttung/research/Monash_CCU/mock_data/visual_data/format_mp4_video'

for file_name in os.listdir(path_video_folder_ldcc_in):
    full_file_path_in = osp.join(path_video_folder_ldcc_in, file_name)

    file_name_no_ext = file_name.split('.')[0]
    full_file_path_out = osp.join(path_video_out_folder_mp4, file_name_no_ext+'.mp4')

    bashCommand = "dd if=%s bs=1024 skip=1 of=%s"%(full_file_path_in, full_file_path_out)
    os.system(bashCommand)
