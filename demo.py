import torch
import clip
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torchvision.models as models

class CLIPMultiLabel(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPMultiLabel, self).__init__()
        self.clip_model = clip_model
        self.fc = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
        image_features = image_features.float()  # 转换为 float32 类型
        return self.fc(image_features)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.features = nn.ModuleList(base)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [32, 270, 480]
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [64, 135, 240]
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # [128, 69, 120]
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # [256, 35, 60]
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # [256, 18, 30]
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # [256, 9, 15]
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*9*15, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 6, bias=False)
        )

        self.last = nn.Sigmoid()
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
        x = self.last(x)
        return x

Label_Map = {
    '流动': 0,
    '混乱': 1,
    '整合': 2,
    '分裂': 3,
    '空洞': 4,
    '联结': 5
}
    
if __name__ == "__main__":
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 6  # 修改为你的实际标签数量

     ######## CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = CLIPMultiLabel(model, num_classes)


    #### ResNet34
    # model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, 512),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(512, num_classes),
    #     nn.Sigmoid()
    # )

    # model = ConvNet()    #### 自定义网络
    

    model = model.to(device)

    ### 加载训练好的模型参数

    model.load_state_dict(torch.load("D:\Vscode_workspace\projects\Sandbox_Theme\checkpoint\clip_epoch_60.pth", map_location=device))
    # model.load_state_dict(torch.load("D:\Vscode_workspace\projects\Sandbox_Theme\checkpoint\\resnet_epoch_40.pth", map_location=device))
    # model.load_state_dict(torch.load("D:\Vscode_workspace\projects\Sandbox_Theme\checkpoint\\cnn_epoch_95.pth", map_location=device))

    model.eval()

    # 定义预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ### 自定义的网络没有进行resize
    transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 测试单张图片
    image_path = "D:\Vscode_workspace\projects\Sandbox_Theme\\resource\homework_sand_label_datasets\\20210428190921_1235\BireView.png"
    # image_path = "D:\Vscode_workspace\projects\Sandbox_Theme\\resource\homework_sand_label_datasets\\20201130095147_72\BireView.png"
    image = Image.open(image_path)

    #### 使用resnet34和clip时用这个transform
    image = transform(image).unsqueeze(0).to(device)  # 添加batch维度并转移到GPU

    #### 使用自定义的网络时用这个transform
    # image = transform_cnn(image).unsqueeze(0).to(device)  # 添加batch维度并转移到GPU


    with torch.no_grad():
        outputs = model(image)
        predictions = (outputs > 0.5).float()

    # 打印结果
    print("Predictions:", predictions.cpu().numpy())
    print("Labels:", [key for i,e in enumerate(predictions.cpu().numpy()[0]) if e == 1 for key,val in Label_Map.items() if val==i])
 