import os
import torch

class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, feat_path: str, mask_path: str):
        self.img_feat_path = feat_path
        self.img_feat_mask = None
        if os.path.exists(mask_path):
            self.img_feat_mask = torch.load(mask_path)

        self.size = len([f for f in os.listdir(self.img_feat_path) if not f.startswith(".")])

    def __getitem__(self, idx):
        one_feat_path=os.path.join(self.img_feat_path, f"{idx+1}.pth")
        one_feat=torch.load(one_feat_path)[0]
        if self.img_feat_mask is None:
            return one_feat, None
        else:
            return one_feat, self.img_feat_mask[idx]

    def __len__(self):
        return self.size
