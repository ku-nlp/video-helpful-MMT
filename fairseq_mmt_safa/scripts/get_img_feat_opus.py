# please see scripts/README.md firstly. 
# python3 ~/code/fairseq_mmt_safa/scripts/get_img_feat_opus.py --model vit_base_patch16_384 --ipath /OpusEJ_central_frame --opath /OpusEJ_central_frame_vit_feature
import timm
import os
import torch
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import argparse
from PIL import Image

dic_model = [
    'vit_tiny_patch16_384',
    'vit_small_patch16_384',
    'vit_base_patch16_384',
    'vit_large_patch16_384',
]

if __name__ == "__main__":
    # please see scripts/README.md firstly. 
    gpu=1
    parser = argparse.ArgumentParser(description='which dataset')
    parser.add_argument('--ipath', type=str)
    parser.add_argument('--opath', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    # /path/flickr30k
    input_path = args.ipath
    model_name = args.model
    save_dir = args.opath
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   
    print('extracting image feature from '+model_name) 
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(f'cuda:{gpu}') # if use cpu, uncomment '.to('cuda:0')'
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    count = 1

    filenames = os.listdir(input_path)
    
    res_shape=[]
    
    with torch.no_grad():
        for filename in tqdm(filenames):
            path=os.path.join(input_path, filename)
            if path.endswith(".pth"):
                os.remove(path)
            else:
                img = Image.open(path).convert("RGB")
                input = transform(img).unsqueeze(0).to(f'cuda:{gpu}') # transform and add batch dimension

                out = model.forward_features(input)

                feat = out.detach().cpu()
                torch.save(feat, os.path.join(save_dir, filename.split(".")[0]+'.pth'))
                count += 1
    
    print(f'feature shape:{feat.shape}', ',save in:', save_dir)
        
        
