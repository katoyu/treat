from arguments import parser

import torch
from torch import nn

import numpy as np
import pickle
import os
import cv2

from torch.utils.data import DataLoader, SubsetRandomSampler
from models import MyData

from models import MultiTask
import sys
import os.path
import matplotlib.pyplot as plt

from logger import get_logger

logger = get_logger(__name__)

IMG_SIZE = 224

# np.random.seed(0)
# torch.manual_seed(0)

args = parser.parse_args()
args.data = args.data.split(',')
args.data.sort()

learning_rate = 0.0001
num_epochs = 100

# モデルの定義
model = MultiTask(args)

if torch.cuda.is_available() == True:
    logger.info("Using GPU")
    model.cuda()
else:
    logger.info("No using GPU")

# lossの設定
rcriterion = torch.nn.MSELoss(reduction='none')
ccriterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
)

# args, save, splits読み込み
# 学習モデルのために何の情報が必要？→モデルの重み
# job_0_model_alexnet_alpha_0.25_datalimit_40250.pth
model_path = '/content/drive/MyDrive/Colab notebooks/Logo-generation/Trick_or_TReAT/treat/save/job_0_model_alexnet_alpha_0.25_datalimit_40250.pth'
model.load_state_dict(torch.load(model_path))
 
# 異なるラベルの二つの適当な画像読み込み
# Initialize train dataset
logger.info("Starting Loading data")
dataset = MyData(args, img_size=IMG_SIZE)
print('----------------')
print(len(dataset))
print('----------------')
print(len(dataset[0]))
print('----------------')
print(dataset[0][0].shape)
print(dataset[0][1])
print(dataset[0][2])
print(dataset[0][3])

print('----------------')

# dataset_in = dataset[0][0].detach().cpu().numpy().transpose(1,2,0)
dataset_in = dataset[0][0].permute(1,2,0).detach().cpu().numpy()
# dataset_in = dataset[0][0].permute(1,2,0)
dataset_in = cv2.cvtColor(dataset_in, cv2.COLOR_RGB2BGR)
dataset_in = dataset_in*0.5
dataset_in = dataset_in+0.5
dataset_in = dataset_in*255
cv2.imwrite("image_dataset.png", dataset_in)

dataset_len = len(dataset)
logger.info("Dataset obtained successfully")

# Split data as train and test
logger.info("Starting data splitting")


# If indices are already saved then load from file directly else perform split once and save
tv_split = int(np.floor(0.1 * len(dataset)))

split_name = "%s-%d-%d" % ("-".join(args.data), args.datalimit, len(dataset))
if os.path.isfile('splits/split-%s.pkl' % split_name):
    logger.info("Using saved split splits/split-%d.pkl" % len(dataset))
    with open('splits/split-%s.pkl' % split_name, 'rb') as f:
        myindices = pickle.load(f)
    train_idx, val_idx = myindices
else:
    logger.info("Splitting data for the first time")
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[tv_split:], indices[:tv_split]
    myindices = [train_idx, val_idx]
    with open('splits/split-%s.pkl' % split_name, 'wb') as f:
        pickle.dump(myindices, f)

print('train_idx, val_idx', len(train_idx), len(val_idx))


train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

dataloader_train = DataLoader(dataset, batch_size=args.batchsize, sampler=train_sampler, num_workers=1)
dataloader_val = DataLoader(dataset, batch_size=args.batchsize, sampler=val_sampler, num_workers=1)

loss_total_train = lr_total_train = lc_total_train = 0
total_letters = 0

routput_list = []
for data in dataloader_train:
        img, labels, weights, identity = data
        if torch.cuda.is_available() == True:
            img = img.float().cuda()
            labels = labels.long().cuda()
            weights = weights.float().cuda()
            identity = identity.cuda()
        else:
            img = img.float()
            labels = labels.long()
            weights = weights.float()
            identity = identity
        # print('img[0]', img[0].reshape([224, 224, 3]).shape)
        img_in = img[0].permute(1,2,0).detach().cpu().numpy()
        # print(img_in)
        img_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)
        img_in = img_in*0.5
        # print(img_in)
        img_in = img_in+0.5
        # print(img_in)
        img_in = img_in*255
        # print(img_in)

        cv2.imwrite("image_in.png", img_in)

        # Markers for letters which are used for classification loss
        letter_flags = (identity == 2).float()
        total_letters += letter_flags.sum()
        # ===================forward=====================
        # print('img', img.shape)
        routput, coutput, _ = model(img)
        print(routput[0].shape, type(routput))
        routput_list.append(routput[0])

tensor  = routput_list[0].permute(1,2,0).detach().cpu().numpy()
tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
tensor = tensor*0.5
tensor = tensor+0.5
tensor = tensor*255
cv2.imwrite("image.png", tensor)