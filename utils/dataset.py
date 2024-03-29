from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os


transform=transforms.Compose({
    # 转化为Tensor
    transforms.ToTensor()
})

# 首先继承Dataset写一个对于数据进行读入和处理的方式
class MyDataset(Dataset):
    def __init__(self, path):
        self.mode=('train' if 'mask' in os.listdir(path) else 'test')#表示训练模式
        self.path=path # 图片路径
        dirlist=os.listdir(path+'image-2/')  # 图片的名称
        self.name=[n for n in dirlist if n[-3:]=='png'] #只读取图片
        
    def __len__(self):
        return len(self.name)
    
    def __getitem__(self,index):  # 获取数据的处理方式
        name=self.name[index]
        # 读取原始图片和标签
        if self.mode=='train':  # 训练模式
            ori_img=cv2.imread(self.path+'image/'+name)  # 原始图片
            lb_img=cv2.imread(self.path+'mask/'+name)  # 标签图片
            ori_img=cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)  # 转为RGB三通道图
            lb_img=cv2.cvtColor(lb_img,cv2.COLOR_BGR2GRAY)  # 掩膜转为灰度图
            return transform(ori_img),transform(lb_img)
        
        if self.mode=='test':  # 测试模式
            ori_img=cv2.imread(self.path+'image-2/'+name)  # 原始图片
            ori_img=cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)  # 转为RGB三通道图
            return transform(ori_img)


if __name__ == '__main__':
    # 加载数据集
    train_path='/work/share/qinyf/teeth/train/'
    traindata=MyDataset(train_path)
    print(len(traindata))