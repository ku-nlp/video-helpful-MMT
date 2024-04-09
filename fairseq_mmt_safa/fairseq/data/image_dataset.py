import os
import torch
import numpy as np
import random
import json
 
# For fairseq_mmt_safa_video
class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, feat_path: str, split:str, mask_path: str, image_feat_dim: list):
        #######################
        video_list_path=os.path.join("/home/code/fairseq_mmt_safa/data/opus-ja-en", split+".video") 
        ######################
        videos=[]
        with open(video_list_path, 'r') as f:
            videos=[line.rstrip() for line in f.readlines()]
        # features=[os.path.join(feat_path, "i3d_resnet50_v1_kinetics400_"+video+"_feat.npy") for video in videos] # i3d_resnet50_v1_kinetics400_opus_1714206_123.mp4_feat.npy
        self.image_feat_dim=image_feat_dim
        
        #######################
        self.return_dist=True
#         if self.return_dist:
#             print("data_loader return average distance")
#             path="/home/code/dup_crowdsourcing/data/average_dist_video-dist.json"
#             with open(path, 'r') as f:
#                 self.average_dist_dict=json.load(f)
#             self.average_dists=[]
        
        if self.return_dist:
            print("data_loader return distance as 2")
            path="/home/code/dup_crowdsourcing/data/average_dist_video-dist.json"
            with open(path, 'r') as f:
                self.average_dist_dict=json.load(f)
            for video in self.average_dist_dict:
                self.average_dist_dict[video]=2
            self.average_dists=[]
        #######################
        
        # check whether feature files exist
        features=[]
        no_exist_count=0
        exist_features=set(os.listdir(feat_path))
        for video in videos:
            if "i3d" in feat_path:
                feature="i3d_resnet50_v1_kinetics400_"+video+"_feat.npy"
            elif ("videoMAE" in feat_path) or ("c4c" in feat_path) or ("detr" in feat_path):
                feature=video.replace(".mp4", ".pth")
            else:
                print("wrong data type")
                raise
            if feature in exist_features:
                features.append(os.path.join(feat_path, feature))
                if self.return_dist:
                    if video in self.average_dist_dict:
                        self.average_dists.append(self.average_dist_dict[video])
#                         print(self.average_dist_dict[video])
                    else:
                        self.average_dists.append(0)
            else:
                features.append(None)
                no_exist_count+=1
                self.average_dists.append(0)
        print(f"{no_exist_count} features not exist!")
        
        #######################
#         random.Random(4).shuffle(features)
#         random.shuffle(features)
#         print("Randomly use features\nRandomly use features\nRandomly use features")
        #######################
        self.img_feat_paths = features
        self.img_feat_mask = None
        if os.path.exists(mask_path):
            self.img_feat_mask = torch.load(mask_path)

        self.size = len(self.img_feat_paths)
        

    def __getitem__(self, idx):
        one_feat_path=self.img_feat_paths[idx]
        if one_feat_path:
            if one_feat_path.endswith(".pth"):
                one_feat=torch.load(one_feat_path)
                one_feat=torch.from_numpy(one_feat)
            elif one_feat_path.endswith(".npy"):
                one_feat=np.load(one_feat_path)
                one_feat=torch.from_numpy(one_feat)
            else:
                raise 
            one_feat=torch.squeeze(one_feat)
        else:
#             print(one_feat_path, "not found")
            one_feat=torch.zeros(self.image_feat_dim)
        if self.return_dist:
            return one_feat, None, torch.tensor(self.average_dists[idx])
        else:
            return one_feat, None, torch.tensor(0)

    def __len__(self):
        return self.size
