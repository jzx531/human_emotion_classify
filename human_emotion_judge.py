import MyData
import torch
from PIL import Image
import cv2
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
cut_size=48
transform1 = transforms.Compose([
    transforms.Resize((cut_size,cut_size)) # 统一图片大小为224 x 224
    # other transformations...
])
transform2=transforms.Compose([transforms.ToTensor()])



class my_Dataset(Dataset):
    def __init__(self):
        # 创建5*2的数据集
        self.data = torch.tensor([[1, 2], [3, 4], [2, 1], [3, 4], [4, 5]])
        # 5个数据的标签
        self.label = torch.tensor([0, 1, 0, 1, 2])

    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index], self.label[index]  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)


def zip_image(datasets_train):
    path = datasets_train[0][0]
    img1 = path
    img1 = transform1(img1)
    img1 = transform2(img1)
    img1 = torch.unsqueeze(img1, dim=0).type(torch.FloatTensor)
    size = img1.size()
    for i in range(len(datasets_train) - 1):
        path = datasets_train[i + 1][0]
        img = path
        img = transform1(img)
        img = transform2(img)
        img = torch.unsqueeze(img, dim=0).type(torch.FloatTensor)
        if img.size() == size:
            img1 = torch.cat([img1, img], dim=0)
    return img1

class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv0 = nn.Sequential(
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=32,  # n_filters 卷积核的高度
                kernel_size=3,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=32,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=64,  # n_filters 卷积核的高度
                kernel_size=3,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        self.conv2=nn.Sequential(# 输入图像大小(16,14,14)
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=64,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=128,  # n_filters 卷积核的高度
                kernel_size=3,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=128,  # n_filters 卷积核的高度
                kernel_size=3,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
           # nn.Flatten(),
        )
        # 建立全卷积连接层
        self.out=nn.Linear(128 * 8 * 8, 7)# 输出是10个类
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        x = self.conv3(x)
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        #output=nn.Softmax(output)
        return output

cnn = CNN()
#加载模型，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
cnn.load_state_dict(torch.load('emotions.pkl'))
print("module load successfully")
cnn.eval()


torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同
# 超参数
EPOCH = 2  # 训练整批数据的次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False


# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

label_dir_list=["angry","disgust","fear","happy","neutral","sad","surprise"]
root_dir_train="train"
#读取tensor——train
e_t=torch.load('tensor{}'.format(label_dir_list[0]))
print(e_t)
label=torch.zeros(e_t.size(0),dtype=torch.long)
for i in range(len(label_dir_list)-1):
    img=torch.load('tensor{}'.format(label_dir_list[i+1]))
    e_t = torch.cat([e_t, img], dim=0)
    img_label=torch.ones(img.size(0),dtype=torch.long)*(i+1)
    label=torch.cat([label,img_label],dim=0)
print("tensor load successfully")
#print(e_t.size())
myDataset=my_Dataset()
myDataset.data=e_t[:28700]
myDataset.label=label[:28700]
#print(myDataset.label.size())
#读取tensor-test：
root_dir_test="test"
#读取tensor——train
tr_t=torch.load('tensor_test{}'.format(label_dir_list[0]))
print(e_t)
label=torch.zeros(tr_t.size(0),dtype=torch.long)
for i in range(len(label_dir_list)-1):
    img=torch.load('tensor_test{}'.format(label_dir_list[i+1]))
    tr_t = torch.cat([tr_t, img], dim=0)
    img_label=torch.ones(img.size(0),dtype=torch.long)*(i+1)
    label=torch.cat([label,img_label],dim=0)
print("tensor load successfully")
#print(e_t.size())
mytrainDataset=my_Dataset()
mytrainDataset.data=tr_t
mytrainDataset.label=label


from torch.utils.data import DataLoader
data = myDataset
print(data.data.size())
print(data.label.size())
my_loader = DataLoader(data,batch_size=50,shuffle=True,num_workers = 0,drop_last=True)
train=True
if(train):
   for epoch in range(EPOCH):
     for step, (b_x, b_y) in enumerate(my_loader):  # 分配batch data
         #print(b_x)
         #print(b_y)
         output = cnn(b_x)  # 先将数据放到cnn中计算output
         loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
         optimizer.zero_grad()  # 清除之前学到的梯度的参数
         loss.backward()  # 反向传播，计算梯度
         optimizer.step()  # 应用梯度

         if step % 50 == 0:
             print('Epoch: ', epoch, 'trainover')
             """
             test_output = cnn(mytrainDataset.data)
             pred_y = torch.max(test_output, 1)[1].data.numpy()
             accuracy = float((pred_y == mytrainDataset.label.data.numpy()).astype(int).sum()) / float(mytrainDataset.label.size(0))
             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
             """
   torch.save(cnn.state_dict(), 'emotions.pkl')#保存模型
   print("module save successfully")


emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png"
,3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}

test_output = cnn(mytrainDataset.data)
pred_y = torch.max(test_output, 1)[1].data.numpy()
accuracy = float((pred_y == mytrainDataset.label.data.numpy()).astype(int).sum()) / float(mytrainDataset.label.size(0))
print( accuracy)

"""
#当需要转换图片变为tensor时进行下列程序
data_train=list()
img_train=list()
for i in range(len(label_dir_list)):
    data_train.append(MyData.MyData(root_dir_train,label_dir_list[i]))
    e_t=zip_image(MyData.MyData(root_dir_train,label_dir_list[i]))
    img_train.append(e_t)
    torch.save(e_t,'tensor{}'.format(label_dir_list[i]))#保存转化过的张量
    print("img_tensor save successfully")
for j in range(len(label_dir_list)):
     print(img_train[j].size())
"""
"""
#读取数据
img_reader=list()
for i in range(len(label_dir_list)):
    img=torch.load('tensor{}'.format(label_dir_list[i]))
    img_reader.append(img)
    print("tensor load successfully")
for j in range(len(label_dir_list)):
     print(img_reader[j].size())


train=True
if(train):
   step=0
   for epoch in range(EPOCH):
     for i in range(len(label_dir_list)):
        for j in range(5):#img_reader[i].size(0)):
            img_train = img_reader[i]
            output = cnn(img_train)
            #print(img_train.size())
            loss = loss_func(output[j], torch.tensor(i,requires_grad=False))  # 输出和真实标签的loss，二者位置不可颠倒
            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()
            step+=1
            if step % 5 == 0:
                print('Epoch: ', epoch, 'trainover')
"""