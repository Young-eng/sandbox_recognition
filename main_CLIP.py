#coding = gbk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import clip
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import argparse
import numpy as np
import json
import os

parser = argparse.ArgumentParser(
    description='Sandbox Theme Recognition Training With Pytorch')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--epoch', default=100, type=int,
                    help='Batch size for training')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--num_classes', type=int, default=6,
                    help='The number of Label classes')
args = parser.parse_args()


class MyDataSet(Dataset):
    def __init__(self, data, label,transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.length = data.shape[0]

    def __getitem__(self, idnex):
        data = self.data[idnex]
        label = self.label[idnex]
        if self.transform is None:
            return data, label
        else:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return self.length

Label_Map = {
    '流动': 0,
    '混乱': 1,
    '整合': 2,
    '分裂': 3,
    '空洞': 4,
    '联结': 5
}


def get_filelist(dir, Imglist, labellist,transform):
    if os.path.isdir(dir):
        for s in os.listdir(dir):
            current_dir = os.path.join(dir, s)
            for f in os.listdir(current_dir):
                file = os.path.join(current_dir, f)
                # print(file)
                if file.split('.')[0].split('/')[-1] == 'BireView':
                    img = Image.open(file)
                    img = transform(img)
                    Imglist.append(img)
                elif file.split('.')[0].split('/')[-1].startswith('202'):
                    tmptensor = torch.zeros(6)
                    indexlist = []
                    
                    with open(file, 'r', encoding="utf-8") as f:
                        data = json.load(f)
                        for theme in data["themes"]: # 一个list
                            assert Label_Map[theme["name"]] == theme["type"]
                            indexlist.append(theme["type"])
                        tmptensor[indexlist] = 1
                        labellist.append(tmptensor)
                        tmptensor = 0
    
    return Imglist, labellist


  # 修改 CLIP 模型的图像编码器以适应多标签分类任务
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
    
    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        image_features.float()
        return self.fc(image_features)
    

if __name__=='__main__':

    root = os.path.dirname(__file__)
    train_root = os.path.join(root, "dataset/train")
    test_root = os.path.join(root, "dataset/val")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    X_train, Y_train = get_filelist(train_root, [], [],transform=transform)
    X_train = torch.stack(X_train, dim=0)
    Y_train = torch.stack(Y_train, dim=0)
    train_set = MyDataSet(data=X_train, label=Y_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    X_test, Y_test = get_filelist(test_root, [], [],transform=transform)
    X_test = torch.stack(X_test, dim=0)
    Y_test = torch.stack(Y_test, dim=0)
    test_set = MyDataSet(data=X_test, label=Y_test)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    writer = SummaryWriter('./runs_clip')

    # 修改最后的全连接层
    num_classes = args.num_classes

    # 加载预训练的 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    model = CLIPMultiLabel(model, num_classes)
    model = model.to(device).float()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    # 训练模型
    num_epochs = args.epoch

    Precision_num = []

    threshold = 0.5

    for epoch in range(num_epochs):
        model.train()
        running_loss,val_loss = 0.0,0.0

        t_preds = []
        t_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            t_preds.append(outputs.detach().cpu())
            t_labels.append(labels.detach().cpu())
        
        t_preds = torch.cat(t_preds)
        t_labels = torch.cat(t_labels)
        t_preds = (t_preds > threshold).float()

        writer.add_scalar('Training Loss', running_loss/args.batch_size, epoch + 1)
        train_precision = precision_score(t_labels, t_preds, average='samples')

        writer.add_scalar('Train_Precision', train_precision, epoch+1)
        
        # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}],Train Loss: {running_loss/len(train_loader):.4f} Train Precision: {train_precision:.4f}')

    # 评估模型
        if epoch % 5 == 4 and epoch > 20:
            print('Saving epoch %d model ...' % (epoch + 1))
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(), './checkpoint/clip_epoch_%d.pth' % (epoch + 1))
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    all_preds.append(outputs.cpu())
                    all_labels.append(labels.cpu())

                    v_loss = criterion(outputs, labels)
                    val_loss += v_loss.item()

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            scheduler.step(val_loss)

            # threshold = 0.5
            all_preds = (all_preds > threshold).float()

            precision = precision_score(all_labels, all_preds, average='samples')


            print("Precision of the network on the 49 test images:%.2f %%" % (100 * precision))    
            # print('[epoch: %d]  val loss: %.3f' % (epoch + 1, val_loss))
            print(f'Epoch [{epoch+1}/{num_epochs}],val Loss: {val_loss/len(test_loader):.4f} val Precision: {precision:.4f}')
            
            writer.add_scalar('Val_Loss', val_loss/len(test_loader), epoch+1)
            writer.add_scalar('Val_Precision', precision, epoch+1)
            print("===============================================")
  
            Precision_num.append(precision)
    print("best Precision is %.2f, corresponding epoch is %d" % (max(Precision_num), (np.argmax(Precision_num) + 1) * 5 + 20))
    writer.close()
