import torch
from unet.unet_model import UNet
from utils.dataset import MyDataset
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


#推荐使用gpu进行训练
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
print('Loading UNet ......')

model = UNet(n_channels=3, n_classes=1, bilinear=False)
#加载模型到gpu或cpu
model.to(device)
print("Loading UNet successfully !")

# pretrains
print('UNet loading pretrained ......')
state_dict = torch.load('weights/unet_carvana_scale1.0_epoch2.pth', map_location=device)
# model.load_state_dict(state_dict)
new_state_dict = {}
for key, value in state_dict.items():
    # print(key, value.shape)
    if key.split('.')[0] == 'outc':
        continue
    new_state_dict[key]=value
# new_state_dict = {k: v for k, v in pretrained_weights.items() if 'fc' not in k}
model.load_state_dict(new_state_dict, strict=False)
print('UNet loading pretrained successfully !')

# dataset
print('Loading test dataset ......')
test_path=r'E:\other\teeth\\'
testdata=MyDataset(test_path)
print("Loading test dataset successfully !")

img_save_path = r'E:\other\teeth\infer-2_npreotrain\\'
os.makedirs(img_save_path, exist_ok=True)
for i,inputs in tqdm(enumerate(testdata)):
    #原始图片和标签
    inputs=inputs.to(device)
    # 输出生成的图像
    out = model(inputs.view(1,3,320,640)) # 模型预测
    #对输出的图像进行后处理
    threshold=0.5
    out= torch.where(out >=threshold, torch.tensor(255,dtype=torch.float).to(device),out)
    out= torch.where(out < threshold, torch.tensor(0,dtype=torch.float).to(device),out)
    #保存图像
    out= out.detach().cpu().numpy().reshape(1,320,640)
    #注意保存为1位图提交
    img = Image.fromarray(out[0].astype(np.uint8))
    img = img.convert('1')
    img.save(img_save_path + testdata.name[i])
