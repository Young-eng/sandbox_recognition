import os
import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import time
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import cv2
import json
import copy
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score,precision_score

parser = argparse.ArgumentParser(
    description='VGG16 for classification in cifar10 Training With Pytorch')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
                    
parser.add_argument('--epoch', default=100, type=int,
                    help='epoch for training')
                    
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                    help='yes or no to choose using warmup strategy to train')
parser.add_argument('--wp_epoch', type=int, default=3,
                    help='The upper bound of warm-up')
parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
args = parser.parse_args()


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
        x = self.classifier(x)  
        x = self.last(x)
        return x


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Label_Map = {
    '流动': 0,
    '混乱': 1,
    '整合': 2,
    '分裂': 3,
    '空洞': 4,
    '联结': 5
}


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


def get_filelist(dir, Imglist, labellist,transform):
    if os.path.isdir(dir):
        for s in os.listdir(dir):
            current_dir = os.path.join(dir, s)
            for f in os.listdir(current_dir):
                file = os.path.join(current_dir, f)
                # print(file)
                if file.split('.')[0].split('/')[-1] == 'BireView':
                    # img = copy.deepcopy(cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1))
                    img = Image.open(file)
                    # print(img.shape)
                    img = transform(img)
                    # img = torch.from_numpy(img)
                    # img = img.permute(2, 0, 1).float()
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



if __name__ == '__main__':
    net = ConvNet()
    writer = SummaryWriter('./runs')

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    root = os.path.dirname(__file__)
    train_root = os.path.join(root, "dataset/train")
    test_root = os.path.join(root, "dataset/val")

    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    X_train, Y_train = get_filelist(train_root, [], [],transform=transform_train)
    X_train = torch.stack(X_train, dim=0)
    Y_train = torch.stack(Y_train, dim=0)
    train_set = MyDataSet(data=X_train, label=Y_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    X_test, Y_test = get_filelist(test_root, [], [],transform=transform_val)
    X_test = torch.stack(X_test, dim=0)
    Y_test = torch.stack(Y_test, dim=0)
    test_set = MyDataSet(data=X_test, label=Y_test)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    epoch_size = args.epoch
    base_lr = args.lr
    criterion = nn.BCELoss()  # 定义损失函数：交叉熵
    acc = []
    start = time.time()
    losslist = []

    threshold = 0.5
    

    for epoch in range(epoch_size):
        train_loss,val_loss = 0.0,0.0

        t_preds = []
        t_labels = []

        # 使用阶梯学习率衰减策略
        if epoch in [30]:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (inputs, labels) in enumerate(train_loader, 0):
            net.train()
            # 使用warm-up策略来调整早期的学习率
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)


            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            
            loss = criterion(outputs, labels).cuda()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            t_preds.append(outputs.detach().cpu())
            t_labels.append(labels.detach().cpu())
        
        t_preds = torch.cat(t_preds)
        t_labels = torch.cat(t_labels)
        t_preds = (t_preds > threshold).float()

        print('[epoch: %d]  Train loss: %.3f' % (epoch + 1, train_loss / args.batch_size))

        losslist.append(train_loss)
        lr_1 = optimizer.param_groups[0]['lr']
        print("learn_rate:%.15f" % lr_1)
        
        writer.add_scalar('Train_Loss', train_loss / args.batch_size, epoch+1)
        train_precision = precision_score(t_labels, t_preds, average='samples')

        writer.add_scalar('Train_Precision', train_precision, epoch+1)

        print(f'Epoch [{epoch+1}/{epoch_size}],Train Loss: {train_loss/len(train_loader):.4f} Train Precision: {train_precision:.4f}')

        if epoch % 5 == 4 and epoch > 20:
            print('Saving epoch %d model ...' % (epoch + 1))
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(net.state_dict(), './checkpoint/cnn_epoch_%d.pth' % (epoch + 1))

            # 由于训练集不需要梯度更新,于是进入测试模式
            net.eval()
            correct = 0.0
            total = 0
            
            all_preds = []
            all_labels = []
            with torch.no_grad():  # 训练集不需要反向传播
                print("=======================validation=======================")
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    pred = (net(inputs) > 0.5).int()

                    v_loss = criterion(net(inputs), labels).cuda()
                    
                    val_loss += v_loss.item()
                    all_preds.append(net(inputs).cpu())
                    all_labels.append(labels.cpu())
                    

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            thr = 0.5
            all_preds = (all_preds > thr).float()

            # print(all_labels)
            # print(all_preds)
            precision = precision_score(all_labels, all_preds, average='samples')
    
            
            print("Precision of the network on the 49 test images:%.2f %%" % (100 * precision))
            print('[epoch: %d]  val loss: %.3f' % (epoch + 1, val_loss))
            
            writer.add_scalar('Val_Loss', val_loss/len(test_loader), epoch+1)
            writer.add_scalar('Val_Precision', precision, epoch+1)
            print("===============================================")

            acc.append(100 * precision)
            net.train()
    print("best Val Precision is %.2f, corresponding epoch is %d" % (max(acc), (np.argmax(acc) + 1) * 5 + 20))
    print("===============================================")
    end = time.time()
    print("time:{}".format(end - start))
    writer.close()

