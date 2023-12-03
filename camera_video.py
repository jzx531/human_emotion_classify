import cv2
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
face_detect=cv2.CascadeClassifier(r"F:\DQN_cartpole\haarcascade_frontalface_alt.xml")
cut_size=48
transform1 = transforms.Compose([
    transforms.Resize((cut_size,cut_size)) # 统一图片大小为224 x 224
    # other transformations...
])
transform2=transforms.Compose([transforms.ToTensor()])

class FaceCNN(nn.Module):
     # 初始化网络结构
     def __init__(self):
         super(FaceCNN, self).__init__()

         # 第一次卷积、池化
         self.conv1 = nn.Sequential(
             # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，卷积核大小kernel_size，步长stride，对称填0行列数padding
             # input:(bitch_size, 1, 48, 48), output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # 卷积层
             nn.BatchNorm2d(num_features=64), # 归一化
             nn.RReLU(inplace=True), # 激活函数
             # output(bitch_size, 64, 24, 24)
             nn.MaxPool2d(kernel_size=2, stride=2), # 最大值池化
         )

         # 第二次卷积、池化
         self.conv2 = nn.Sequential(
             # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 24, 24), (24-3+2*1)/1+1 = 24
             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(num_features=128),
             nn.RReLU(inplace=True),
             # output:(bitch_size, 128, 12 ,12)
             nn.MaxPool2d(kernel_size=2, stride=2),
         )

         # 第三次卷积、池化
         self.conv3 = nn.Sequential(
             # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12), (12-3+2*1)/1+1 = 12
             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(num_features=256),
             nn.RReLU(inplace=True),
             # output:(bitch_size, 256, 6 ,6)
             nn.MaxPool2d(kernel_size=2, stride=2),
         )

         # 参数初始化
         #self.conv1.apply(gaussian_weights_init)
         #self.conv2.apply(gaussian_weights_init)
         #self.conv3.apply(gaussian_weights_init)

         # 全连接层
         self.fc = nn.Sequential(
             nn.Dropout(p=0.2),
             nn.Linear(in_features=256*6*6, out_features=4096),
             nn.RReLU(inplace=True),
             nn.Dropout(p=0.5),
             nn.Linear(in_features=4096, out_features=1024),
             nn.RReLU(inplace=True),
             nn.Linear(in_features=1024, out_features=256),
             nn.RReLU(inplace=True),
             nn.Linear(in_features=256, out_features=7),
         )

     # 前向传播
     def forward(self, x):
         x = self.conv1(x)
         x = self.conv2(x)
         x = self.conv3(x)
         # 数据扁平化
         x = x.view(x.shape[0], -1)
         y = self.fc(x)
         return y


cnn = FaceCNN()
print(cnn)
cnn.load_state_dict(torch.load('emotions.pkl'))
cnn.eval()
print("neural_net load successfully")
#获取huamn_emotion_judge的cnn_MOdule


emoji_dist = {0: "./emojis/angry.png", 1: "./emojis/disgusted.png", 2: "./emojis/fearful.png"
    , 3: "./emojis/happy.png", 4: "./emojis/neutral.png", 5: "./emojis/sad.png", 6: "./emojis/surpriced.png"}

#对图像进行灰度张量转换
def frame_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=Image.fromarray(np.uint8(gray))
    emotion = transform1(gray)
   # emotion.show()
    emotion_tensor0 = transform2(emotion)
    emotion_tensor = torch.unsqueeze(emotion_tensor0, dim=0).type(torch.FloatTensor)
    return emotion_tensor
def show_cartoon_emoji(out_tensor):
    if out_tensor.size(1)==7:
        index=torch.max(out_tensor,1)[1].item()
    path=emoji_dist[index]
    img=cv2.imread(path)
    return img

#用opencv2对笔记本摄像头进行读取，并进行处理
video=cv2.VideoCapture(0)
#设置宽尺寸:
video.set(propId = 3, value = 96)
#设置高尺寸:
video.set(propId = 4, value = 192)

fps=video.get(cv2.CAP_PROP_FPS)
print(fps)
size=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(size)
while True:
    ret,frame=video.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(
        gray,
        scaleFactor=1.02,
        minNeighbors=6
    )
    if (len(faces)):
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if w > 50 and h > 50:
                crop_img = frame[y:y + h, x:x + w]
                print(crop_img.shape)
    #frame=Image.fromarray(np.uint8(frame))
            emotion_tensor= frame_process(crop_img)
    output=cnn(emotion_tensor)
    #print(output)
    cv2.imshow("video",frame)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    img=show_cartoon_emoji(output)
    a = np.asarray(img)
    cv2.imshow('image', a)
    c=cv2.waitKey(1)
    if c==27:
        break
video.release()
cv2.destroyWindow()


