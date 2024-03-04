from Warehouse import *
from Resnet import *
#创建训练集和测试集
def judge(x,shape):
    flag = 1  # 表示有效
    for i in range(shape):
        if torch.all(x[i] == 0) or torch.all(x[:, i] == 0) or torch.all(x==1) or torch.sum(x)/(shape**2)>0.9:
            flag = 0
            break
    return flag
def creatdata(o_path,shape):
    train_path=os.path.join(o_path,'train_'+str(shape))
    test_path=os.path.join(o_path,'test_'+str(shape))
    trans = transforms.Compose([
        transforms.CenterCrop(400),
        transforms.RandomCrop(shape,shape),
        transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
    ])
    n_train=1
    n_test=1
    for i in range(1,61):
        while n_train<i*1000+1:
            img=trans(Image.open(os.path.join(r'C:\Users\86187\Desktop\2023第一次模拟题(发布)\2023第一次模拟题(发布)\A1题：最强大脑之图形辨识\附件\附件',
                                        'tuxing'+str(i)+'.bmp')).crop([80,35,500,370]))
            img=img.squeeze(0)
            # imgs=rearrange(imgs, 'c  (h h1) (w w2) -> (h w) h1 (w2 c)', h1=shape, w2=shape)
            # for img in imgs:
            if judge(img,shape)==1:
                image = Image.fromarray((np.array(img) * 255).astype(np.uint8))
                if random.random()<1.1:
                    image.save(os.path.join(train_path,str(n_train)+'_'+str(i)+'.bmp'))
                    n_train+=1
                else:
                    image.save(os.path.join(test_path, str(n_test) +'_'+ str(i) + '.bmp'))
                    n_test += 1
    print('生成了{}个训练图像和{}个测试图像'.format(n_train,n_test))

class Dataset(Dataset):
    def __init__(self,o_path,mode,shape):
        super(Dataset, self).__init__()
        self.list=os.listdir(os.path.join(o_path,mode+'_'+str(shape)))
        imgs=[]
        labels=[]
        self.shape=shape
        for i in self.list:
            imgs.append(os.path.join(o_path,mode+'_'+str(shape),i))
            labels.append(str(int(i.split('_')[1].split('.')[0])-1))

        self.img = imgs
        self.label = labels
        self.trans = transforms.Compose([
            transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
        ])
        # self.crop=[80,35,500,370]
    #判断是否满足要求

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        img = Image.open(img)
        # 此时img是PIL.Image类型   label是str类型
        # img=img.crop(self.crop)
        if self.trans is not None:
            img = self.trans(img)
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)

        return img, label


if __name__ == '__main__':

    creatdata('./',20)
    # traindata_40=Dataset('./','train',40)
    # testdata_40=Dataset('./','test',40)
    # traindata_loader = DataLoader(dataset=traindata_40, batch_size=64, shuffle=True)
    # testdata_loader = DataLoader(dataset=testdata_40, batch_size=64, shuffle=True)
    # for i, data in enumerate(traindata_loader):
    #     images, labels = data
    #
    #     # 打印数据集中的图片
    #     img = torchvision.utils.make_grid(images).numpy()
    #     plt.imshow(np.transpose(img, (1, 2, 0)))
    #     plt.show()
    #     print(images.shape)
    #     print(labels)
    #
    #     break



