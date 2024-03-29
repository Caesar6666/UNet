import torch
from torch.utils.data import DataLoader
from unet.unet_model import UNet
from utils.dataset import MyDataset
from utils.loss import dice_loss
from tqdm import tqdm
from thop import profile


#配置模型超参数
#模型保存的路径
model_path='checkpoints/'
#推荐使用gpu进行训练
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#学习率
lr=3e-3
#学习率衰减
weight_decay=1e-5
momentum = 0.999
#批大小
bs=40
#训练轮次
epochs=100
num_workers=28

# model
print('Loading UNet ......')
model = UNet(n_channels=3, n_classes=1, bilinear=False)
# 计算参数量和计算量
inputs = torch.randn(1, 3, 112, 112)
flops, params = profile(model, (inputs,))
print('flops: ', flops, 'params: ', params)

#加载模型到gpu或cpu
model.to(device)
print("Loading UNet successfully !")

# pretrains
print('UNet loading pretrained ......')
state_dict = torch.load('weights/unet_carvana_scale1.0_epoch2.pth', map_location=device)
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
print('Loading train dataset ......')
train_path='/work/share/qinyf/teeth/train/'
traindata=MyDataset(train_path)
print("Loading train dataset successfully !")

# dataloader
#使用traindata创建dataloader对象
print("Creating train dataset ......")
trainloader=DataLoader(traindata,batch_size=bs, shuffle=True, num_workers=num_workers)
print("Creating train dataset successfully !")

#加载优化器,使用Adam,主要是炼的快(๑ت๑)
# optim=torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
optim = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
#使用Binary CrossEntropy作为损失函数，主要处理二分类问题
# BCEloss=nn.BCELoss()


#开始炼丹 没有做验证集，各位可以以自己需要去添加
loss_last=99999
best_model_name='unet'
#记录loss变化
f = open('loss.txt', 'w', encoding='utf-8')
for epoch in range(1,epochs+1):
    for step,(inputs,labels) in tqdm(enumerate(trainloader),desc=f"Epoch {epoch}/{epochs}",
                                       ascii=True, total=len(trainloader)):
        #原始图片和标签
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        loss = dice_loss(out, labels)
        # 后向
        optim.zero_grad()
        #梯度反向传播
        loss.backward()
        optim.step()
    #损失小于上一轮则添加
    if loss<loss_last:
        loss_last=loss
        torch.save(model.state_dict(), model_path+'model_epoch{}_loss{:.3f}.pth'.format(epoch,loss))
        best_model_name=model_path+'model_epoch{}_loss{}.pth'.format(epoch,loss)
    print(f"\nEpoch: {epoch}/{epochs},DiceLoss:{loss}")
    f.write(f"Epoch: {epoch}/{epochs},DiceLoss:{loss}\n")
f.close()
