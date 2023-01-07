import json
import os
import os.path as osp
import pdb

def load_cp_info(path_cp1, path_cp2):
    # read cp
    with open(path_cp1, 'r') as fp:
        cp1 = fp.read().split('\n')

    with open(path_cp2, 'r') as fp:
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

    return cp_video_info

class PairGenerator():
    def __init__(self, cp_video_info, diar_res):
        self.cp_video_info = cp_video_info
        self.diar_res = diar_res

    def gen_vid_pos_and_neg(self):
        pos_pair = {}
        neg_pair = {}

        stat_pair = {'ratio_pair': {}, 'total_pos_pair': 0, 'total_neg_pair': 0}

        for vid_name in self.diar_res:

            diar_res_vid = self.diar_res[vid_name]['SD_list']

            pos_pair_list = []
            neg_pair_list = []
            for idx, cur_SD in enumerate(diar_res_vid):
                if idx < len(diar_res_vid) - 1:
                    # construct positive pairs
                    next_SD = diar_res_vid[idx+1]

                    for Di in cur_SD:
                        for Dk in next_SD:
                            pos_pair_list.append((Di, Dk)) #should we add weight to indicate the frame discrepancy
                # construct negative pairs
                for sub_idx, Di in enumerate(cur_SD[:-1]):
                    # only get the next Dj within similar SD
                    Dj = cur_SD[sub_idx+1]
                    neg_pair_list.append((Di, Dj))


                    # for Dj in cur_SD[sub_idx+1:]:
                    #     neg_pair_list.append((Di, Dj))

            stat_pair['ratio_pair'][vid_name] = str(len(pos_pair_list))+'/'+str(len(neg_pair_list))
            stat_pair['total_pos_pair'] += len(pos_pair_list)
            stat_pair['total_neg_pair'] += len(neg_pair_list)

            pos_pair[vid_name] = pos_pair_list
            neg_pair[vid_name] = neg_pair_list

        return pos_pair, neg_pair, stat_pair
        

if __name__ == "__main__":
    path_diarize_result = '/home/nttung/research/Monash_CCU/CL_Monash/auxiliary_folder/SD_diar_result.json'
    path_cp_all = '/home/nttung/research/Monash_CCU/CL_Monash/auxiliary_folder/joint_cp_info.json'
    
    with open(path_diarize_result, 'r') as fp:
        diar_res = json.load(fp)

    # cp_info
    with open(path_cp_all, 'r') as fp:
        cp_video_info = json.load(fp)


    pair_generator = PairGenerator(cp_video_info, diar_res)
    pos_pair, neg_pair, stat_pair = pair_generator.gen_vid_pos_and_neg()
    pdb.set_trace()


    

