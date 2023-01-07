import os
import numpy as np
import pdb

path_cp_1 = '../../mock_data/annotation/LDC2022E18_CCU_TA1_Mandarin_Chinese_Development_Annotation_V1.0/data/changepoint.tab'
path_cp_2 = '../../mock_data/annotation/LDC2022E18_CCU_TA1_Mandarin_Chinese_Development_Annotation_V2.0/data/changepoint.tab'
path_in_all_download_video = '../../mock_data/all_vid_data/LDC2022E19_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R2_V1.0/tools/dl_tool/uid_list.tab'
path_out_filter_download_video = '../auxiliary_folder/filtered_uid_list.tab'

with open(path_in_all_download_video, 'r') as fp:
    uid_data = fp.read().split('\n')

with open(path_cp_1, 'r') as fp:
    cp1 = fp.read().split('\n')

with open(path_cp_2, 'r') as fp:
    cp2 = fp.read().split('\n')


header = uid_data[0]

cp_video_list = []
for each_line in cp1[1:-1]:
    each_line = each_line.split('\t')
    cp_video_list.append(each_line[1])

for each_line in cp2[1:-1]:
    each_line = each_line.split('\t')
    cp_video_list.append(each_line[1])

###########
filtered_res = ['source_id', 'file_uid', 'url']
filtered_res = ['\t'.join(filtered_res)]

for each_line in uid_data[1:-1]:
    each_info = each_line.split('\t')

    if each_info[1] in cp_video_list:
        tmp = [each_info[0], each_info[1], each_info[2]]
        tmp = '\t'.join(tmp)

        filtered_res.append(tmp)
    
filtered_res = '\n'.join(filtered_res)
with open(path_out_filter_download_video, 'w') as fp:
    fp.write(filtered_res)

pdb.set_trace()