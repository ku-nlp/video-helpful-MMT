# extract image feature via DETR

# python3 ~/code/fairseq_mmt_safa/scripts/get_img_feat_detr_opus.py --ipath /OpusZE_central_frame --opath /OpusZE_central_frame_detr_feature
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import os
import argparse

gpu=3

# running following code, to download DETR offical code and model
# see https://github.com/facebookresearch/detr
model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_dc5', pretrained=True).to(f'cuda:{gpu}')
model.eval()

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='which dataset')
    parser.add_argument('--ipath', type=str)
    parser.add_argument('--opath', type=str)
    args = parser.parse_args()

    input_path = args.ipath
    save_dir = args.opath
    print('extracting image feature from DETR-DC5-R101') 
    
    count = 1
    filenames = os.listdir(input_path)

    # propagate through the model
    with torch.no_grad():
        for filename in tqdm(filenames):
            path = os.path.join(input_path, filename)
            
            img = Image.open(path).convert("RGB")
            # mean-std normalize the input image (batch-size: 1)
            input = transform(img).unsqueeze(0).to(f'cuda:{gpu}') # transform and add batch dimension
            # outputs return 6 decoder layers' features
            # we get the lastest layer's feature
            out = model(input)[-1]
            
            res = out.detach().cpu()
#             print(res.shape)
            torch.save(res, os.path.join(save_dir, filename.split(".")[0]+'.pth'))
            count += 1