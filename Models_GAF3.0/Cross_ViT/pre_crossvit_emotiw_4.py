#----------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------

############在emotiw上预训练cross_vit,,输入变量为特征向量global_features: batch 1 256
#                                                      facefeatures: batch 16 256
#       numclass=3，进行了3分类，而不是256分类的特征向量

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from skimage import io, transform

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision import transforms, utils

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pickle

#---------------------------------------------------------------------------
# IMPORTANT PARAMETERS
#---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else 'cpu'
maxFaces = 16   #原来是15
numClasses = 3

root_dir = "../../Dataset/"
data_dir = '../../Dataset/emotiw/'
batch_size = 32
epochs = 24

#---------------------------------------------------------------------------
# DATASET AND LOADERS
#---------------------------------------------------------------------------
neg_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Negative/'))
neu_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Neutral/'))
pos_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Positive/'))

train_filelist = neg_train + neu_train + pos_train

val_filelist = []
test_filelist = []


with open('../../Dataset/val_list', 'rb') as fp:
    val_filelist = pickle.load(fp)

with open('../../Dataset/test_list', 'rb') as fp:
    test_filelist = pickle.load(fp)

for i in train_filelist:
    if i[0] != 'p' and i[0] != 'n':
        train_filelist.remove(i)

for i in val_filelist:
    if i[0] != 'p' and i[0] != 'n':
        val_filelist.remove(i)

dataset_sizes = [len(train_filelist), len(val_filelist), len(test_filelist)]
print(dataset_sizes)

train_global_data_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),  #原来是224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_global_data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),      #原来是224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_faces_data_transform = transforms.Compose([
        transforms.Resize((96,112)),
        transforms.ToTensor()
    ])

val_faces_data_transform = transforms.Compose([
        transforms.Resize((96,112)),
        transforms.ToTensor()
    ])

class EmotiWDataset(Dataset):

    def __init__(self, filelist, root_dir, loadTrain=True, transformGlobal=transforms.ToTensor(), transformFaces=transforms.ToTensor()):
        """
        Args:
            filelist: List of names of image/feature files.
            root_dir: Dataset directory
            transform (callable, optional): Optional transformer to be applied
                                            on an image sample.
        """

        self.filelist = filelist
        self.root_dir = root_dir
        self.transformGlobal = transformGlobal
        self.transformFaces = transformFaces
        self.loadTrain = loadTrain

    def __len__(self):
        if self.loadTrain:
            return (len(train_filelist))
        else:
            return (len(val_filelist))

    def __getitem__(self, idx):
        train = ''
        if self.loadTrain:
            train = 'train'
        else:
            train = 'val'
        filename = self.filelist[idx].split('.')[0]
        labeldict = {'neg': 'Negative',
                     'neu': 'Neutral',
                     'pos': 'Positive',
                     'Negative': 0,
                     'Neutral': 1,
                     'Positive': 2}

        labelname = labeldict[filename.split('_')[0]]
        image = Image.open(self.root_dir + 'emotiw/' + train + '/' + labelname + '/' + filename + '.jpg')

        image = image.convert( 'RGB')  ##用opencv或者是PIL包下面的图形处理函数，把输入的图片从灰度图转为RGB空间的彩色图。这种方法可以适合数据集中既包含有RGB图片又含有灰度图的情况    后加的

        if self.transformGlobal:
            image = self.transformGlobal(image)

        if image.shape[0] == 1:
            image_1 = np.zeros((3, 224, 224), dtype=float)
            image_1[0] = image
            image_1[1] = image
            image_1[2] = image
            image = image_1
            image = torch.FloatTensor(image.tolist())

        features = np.load(self.root_dir + 'FaceFeatures/' + train + '/' + labelname + '/' + filename + '.npz')['a']
        # features = torch.from_numpy(features)
        numberFaces = features.shape[0]
        maxNumber = min(numberFaces, maxFaces)

        features1 = np.zeros((maxFaces, 3, 96, 112), dtype='float32')
        features2 = np.zeros((maxFaces, 256 ), dtype='float32')

        for i in range(maxNumber):
            face = Image.open(self.root_dir + 'AlignedCroppedImages/' + train + '/' + labelname + '/' + filename + '_' + str(i) + '.jpg')

            if self.transformFaces:
                face = self.transformFaces(face)

            features1[i] = face.numpy()

        #补全人脸个数
        if numberFaces < maxFaces:
            for j in range(maxFaces-numberFaces):
                features1[j+numberFaces] = features1[j]

        features1 = torch.from_numpy(features1)
        # 由于人脸数量不同，所以人脸特征向量维度各不同，同样需要补全或者抛弃一部分。
        for i in range(maxNumber):
            features2[i] = features[i]

        if numberFaces < maxFaces:
            for j in range(maxFaces-numberFaces):
                features2[j+numberFaces] = features2[j]

        features2 = torch.from_numpy(features2)

        features3 = np.load(self.root_dir + 'GlobalFeatures/' + train + '/' + labelname + '/' + filename + '.npz')['a']

        sample = {'image': image, 'features': features1, 'label': labeldict[labelname], 'numberFaces': numberFaces,'face_features': features2,'global_features':features3}

        return sample

train_dataset = EmotiWDataset(train_filelist, root_dir, loadTrain = True, transformGlobal=train_global_data_transform, transformFaces = train_faces_data_transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

val_dataset = EmotiWDataset(val_filelist, root_dir, loadTrain=False, transformGlobal= val_global_data_transform, transformFaces = val_faces_data_transform)

val_dataloader = DataLoader(val_dataset, shuffle =True, batch_size = batch_size, num_workers = 0)

print('Dataset Loaded')

#---------------------------------------------------------------------------
# MODEL DEFINITION
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# CROSS VIT
#---------------------------------------------------------------------------

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = x.to(device)
        self.norm = self.norm.to(device)
        self.fn = self.fn.to(device)
        return self.fn(self.norm(x), **kwargs)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            self.norm =self.norm.to(device)
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)  #32 1 256    32 16 256
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)  #32 1 256    32 1  256
        return sm_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)   #sm_tokens:32 17 256  lg_tokens:32 2 256
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# patch-based image to token embedder

class ImageEmbedder_s(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2       # 应该改为人脸个数？
        patch_dim = 3 *96*112    #3*64^2=12288

        self.to_patch_embedding = nn.Sequential(            # 将批量为 b.通道为 c.高为 h*p1.宽为 w*p2.的图像转化为批量为 b个数为 h*w 维度为  p1*p2*c  的图像块
                                                            # 即，把 b张 c通道的图像分割成 b*（h*w）张大小为 p1*p2*c的图像块
            Rearrange('b p c h w -> b p (c h w) '),         # 例如：(b, c,h*p1, w*p2)->(b, h*w, p1*p2*c)  patch_size为  64  (32, 3, 256, 256)->(32,16,12288)
            nn.Linear(patch_dim, dim),     # 12288 256                     (32, 16, 3, 96, 112) -> (32 ,16,96*112*3=32256)
        )                   #  步骤2patch转化为embedding

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, face_features): #  训练好的脸部特征向量 face_features:  32 16 256

        # x = self.to_patch_embedding(face_features)  # 输出x:32 16 256
        x = face_features
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 被拷贝b次(b是batch的数量)
        cls_tokens = cls_tokens.to(device)

        x = torch.cat((cls_tokens, x), dim=1)    # 添加到patch前面

        pos_embedding = self.pos_embedding
        pos_embedding = pos_embedding.to(device)
        x += pos_embedding[:, :(n + 1)]

        return self.dropout(x)

class ImageEmbedder_l(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )                   #  步骤2patch转化为embedding

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img2): #img2:32 1 256
        x = img2  #x:32 1 256
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens = cls_tokens.to(device)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embedding = self.pos_embedding
        pos_embedding = pos_embedding.to(device)

        x += pos_embedding[:, :(n + 1)]  # 加position embedding

        return self.dropout(x)

# cross ViT class

class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size = 256,
        num_classes = 3,
        sm_dim =256,
        lg_dim=256,

        sm_patch_size = 64,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,

        lg_patch_size = 256,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder_s(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder_l(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        # self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        # self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

        self.sm_mlp_head = nn.LayerNorm(sm_dim)
        self.lg_mlp_head = nn.LayerNorm(lg_dim)

        self.sm_mlp_head_linear = nn.Linear(sm_dim, num_classes)
        self.lg_mlp_head_linear = nn.Linear(lg_dim, num_classes)

        # self.linear = nn.Linear(256, 3)  #后加的，最后的linear层

    def forward(self, face_features,img2):      #  face_features：32 16 256 输入为训练好的人脸特征向量    img2:32 1 256  全局特征向量
        sm_tokens = self.sm_image_embedder(face_features)
        lg_tokens = self.lg_image_embedder(img2)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)   #32 17 256    32 2 256

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))  #sm_cls:32 256   lg_cls:32 256

        self.sm_mlp_head =self.sm_mlp_head.to(device)
        self.lg_mlp_head =self.lg_mlp_head.to(device)

        sm = self.sm_mlp_head(sm_cls)   #多头注意  32 256
        lg = self.lg_mlp_head(lg_cls)   #多头注意  32 256

        sm_logits = self.sm_mlp_head_linear(sm)
        lg_logits = self.lg_mlp_head_linear(lg)

        x = sm_logits + lg_logits  #后加的   x: 32 256
        # x = (self.linear(x))       #后加的，最后的3分类
        return x

#---------------------------------------------------------------------------
# CROSS VIT
#---------------------------------------------------------------------------

model_ft = CrossViT(
    image_size = 256,
    num_classes = 3,
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 256,            # high res dimension     高分辨率尺寸
    sm_patch_size = 64,      # high res patch size (should be smaller than lg_patch_size)

    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 2048,   # high res feedforward dimension   高分辨率前馈维度
    lg_dim = 256,            # low res dimension      低分辨率尺寸
    lg_patch_size = 256,     # low res patch size

    lg_enc_depth = 3,        # low res depth
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0.1,
    emb_dropout = 0.1)
#num_ftrs = model_ft.classifier.in_features
#model_ft.classifier = nn.Linear(num_ftrs, 3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)
model_ft = torch.nn.DataParallel(model_ft)

# print(model_ft)


#---------------------------------------------------------------------------
# TRAINING
#---------------------------------------------------------------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in range(2):
            if phase == 0:
                dataloaders = train_dataloader
                #scheduler.step()改动
                model.train()
            else:
                dataloaders = val_dataloader
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i_batch, sample_batched in enumerate(dataloaders):
                inputs = sample_batched['image']    #inputs:32 3 256 256
                labels = sample_batched['label']    #labels:32
                face_features2 = sample_batched['face_features']
                global_feature = sample_batched['global_features']
                global_feature = global_feature.view(-1, 1, 256)

                inputs = inputs.to(device)
                labels = labels.to(device)
                face_features2 = face_features2.to(device)
                global_feature = global_feature.to(device)

                with torch.set_grad_enabled(phase == 0):   #查一下作用   设置梯度计算开启或关闭的上下文管理器  phase == 0时开启，反之关闭
                    outputs = model(face_features2,global_feature)  # 输入:    face_features2:  32 15 256    inputs:32 3 256 256
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 0:
                        optimizer.zero_grad()  #改动  原来在621行
                        loss.backward()
                        optimizer.step()

                # outputs = model(inputs,inputs)  #改动  outputs:64 3
                # _, preds = torch.max(outputs, 1)
                # loss = criterion(outputs, labels)
                #
                # optimizer.zero_grad()  #改动  原来在621行
                # loss.backward()
                # optimizer.step()
                # #
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format( phase, epoch_loss, epoch_acc))

            if phase == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, '../TrainedModels/TrainDataset/pre_crossvit_emotiw_4')

        scheduler.step()  # 改动
        print()
    time_elapsed = time.time() - since
    print('Training complete in {: .0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  #  原来是lr=0.001   试一下optim.adam原来是SGD
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, eps=1e-08)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)

torch.save(model_ft.state_dict(), "../TrainedModels/pre_crossvit_emotiw_4_state.pt")
torch.save(model_ft, '../TrainedModels/pre_crossvit_emotiw_4')