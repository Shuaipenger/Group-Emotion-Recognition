# IMPORTING MODULES
#----------------------------------------------------------------------------
# coding: utf-8
from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum

#---------------------------------------------------------------------------
# IMPORTANT PARAMETERS
#---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else 'cpu'
root_dir = "../Dataset/"
epochs = 1
batch_size = 8
maxFaces = 15
random_seed = 8

#---------------------------------------------------------------------------
# DATASET AND LOADERS
#---------------------------------------------------------------------------

neg_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Negative/'))
neu_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Neutral/'))
pos_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Positive/'))

train_filelist = neg_train + neu_train + pos_train

val_filelist = []
test_filelist = []

with open('../Dataset/val_list', 'rb') as fp:
    val_filelist = pickle.load(fp)

with open('../Dataset/test_list', 'rb') as fp:
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_global_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_global_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

test_faces_data_transform = transforms.Compose([
        transforms.Resize((96,112)),
        transforms.ToTensor()
    ])

class EmotiWDataset(Dataset):
    
    def __init__(self, filelist, root_dir, loadTrain=True, transformGlobal=transforms.ToTensor(), transformFaces = transforms.ToTensor()):
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
        return (len(self.filelist)) 
 
    def __getitem__(self, idx):
        train = ''
        if self.loadTrain:
            train = 'train'
        else:
            train = 'val'
        filename = self.filelist[idx].split('.')[0]
        labeldict = {'neg':'Negative',
                     'neu':'Neutral',
                     'pos':'Positive',
                     'Negative': 0,
                     'Neutral': 1,
                     'Positive':2}

        labelname = labeldict[filename.split('_')[0]]

        #IMAGE
        image = Image.open(self.root_dir+'emotiw/'+train+'/'+labelname+'/'+filename+'.jpg')
        image = image.convert('RGB')
        if self.transformGlobal:
            image = self.transformGlobal(image)
        if image.shape[0] == 1:
            image_1 = np.zeros((3, 224, 224), dtype = float)
            image_1[0] = image
            image_1[1] = image
            image_1[2] = image
            image = image_1
            image = torch.FloatTensor(image.tolist())

        # skeleton image
        if self.loadTrain:
            train = 'train'
            image2 = Image.open(self.root_dir + 'emotiw_skeletons/' + train + '/' + labelname + '/' + filename + '_rendered.png')
        else:
            train = 'val'
            image2 = Image.open(self.root_dir + 'emotiw_skeletons/' + train + '/' + labelname + '/' + filename + '_rendered.png')
        if self.transformGlobal:
            image2 = self.transformGlobal(image2)
        if image2.shape[0] == 1:
            image_1 = np.zeros((3, 224, 224), dtype=float)
            image_1[0] = image2
            image_1[1] = image2
            image_1[2] = image2
            image2 = image_1
            image2 = torch.FloatTensor(image2.tolist())

        #FEATURES FROM MTCNN

        features = np.load(self.root_dir+'FaceFeatures/'+train+'/'+labelname+'/'+filename+'.npz')['a']
        numberFaces = features.shape[0]
        maxNumber = min(numberFaces, maxFaces)

        features1 = np.zeros((maxFaces, 256), dtype = 'float32')
        for i in range(maxNumber):
            features1[i] = features[i]
        features1 = torch.from_numpy(features1)

        #ALIGNED CROPPED FACE IMAGES

        features2 = np.zeros((maxFaces, 3, 96, 112), dtype = 'float32')
#         print(maxNumber)
        
        for i in range(maxNumber):
            face = Image.open(self.root_dir + 'AlignedCroppedImages/'+train+'/'+ labelname + '/' + filename+ '_' + str(i) + '.jpg')
                
            if self.transformFaces:
                face = self.transformFaces(face)
                
            features2[i] = face.numpy()
            
        features2 = torch.from_numpy(features2)

        #SAMPLE
        sample = {'image': image,'skeleton_image':image2, 'features_mtcnn': features1, 'features_aligned':features2, 'label':labeldict[labelname], 'numberFaces': numberFaces}
        return sample

train_dataset = EmotiWDataset(train_filelist, root_dir, loadTrain = True, transformGlobal=train_global_data_transform, transformFaces=train_faces_data_transform)

train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=0)

val_dataset = EmotiWDataset(val_filelist, root_dir, loadTrain=False, transformGlobal = val_global_data_transform, transformFaces=val_faces_data_transform)

val_dataloader = DataLoader(val_dataset, shuffle =False, batch_size = batch_size, num_workers = 0)

test_dataset = EmotiWDataset(test_filelist, root_dir, loadTrain=False, transformGlobal = test_global_data_transform, transformFaces = test_faces_data_transform)

test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, num_workers = 0)

print('Dataset Loaded')

#---------------------------------------------------------------------------
# SPHEREFACE MODEL FOR ALIGNED MODELS
#---------------------------------------------------------------------------

class LSoftmaxLinear(nn.Module):

    def __init__(self, input_dim, output_dim, margin):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))

        self.divisor = math.pi / self.margin
        self.coeffs = binom(margin, range(0, margin + 1, 2))
        self.cos_exps = range(self.margin, -1, -2)
        self.sin_sq_exps = range(len(self.cos_exps))
        self.signs = [1]
        for i in range(1, len(self.sin_sq_exps)):
            self.signs.append(self.signs[-1] * -1)

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight.data.t())

    def find_k(self, cos):
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            logit = input.matmul(self.weight)
            batch_size = logit.size(0)
            logit_target = logit[range(batch_size), target]
            weight_target_norm = self.weight[:, target].norm(p=2, dim=0)
            input_norm = input.norm(p=2, dim=1)
            norm_target_prod = weight_target_norm * input_norm
            cos_target = logit_target / (norm_target_prod + 1e-10)
            sin_sq_target = 1 - cos_target**2
            
            weight_nontarget_norm = self.weight.norm(p=2, dim=0)
            
            norm_nontarget_prod = torch.zeros((batch_size,numClasses), dtype = torch.float)
            logit2 = torch.zeros((batch_size,numClasses), dtype = torch.float)
            logit3 = torch.zeros((batch_size,numClasses), dtype = torch.float)

            norm_nontarget_prod = norm_nontarget_prod.to(device)
            logit2 = logit2.to(device)
            logit3 = logit3.to(device)
            
            for i in range(numClasses):
                norm_nontarget_prod[:, i] = weight_nontarget_norm[i] * input_norm 
                logit2[:, i] = norm_target_prod / (norm_nontarget_prod[:, i] + 1e-10)
            
            for i in range(batch_size):
                for j in range(numClasses):
                    logit3[i][j] = logit2[i][j] * logit[i][j]

            num_ns = self.margin//2 + 1
            coeffs = Variable(input.data.new(self.coeffs))
            cos_exps = Variable(input.data.new(self.cos_exps))
            sin_sq_exps = Variable(input.data.new(self.sin_sq_exps))
            signs = Variable(input.data.new(self.signs))

            cos_terms = cos_target.unsqueeze(1) ** cos_exps.unsqueeze(0)
            sin_sq_terms = (sin_sq_target.unsqueeze(1)
                            ** sin_sq_exps.unsqueeze(0))

            cosm_terms = (signs.unsqueeze(0) * coeffs.unsqueeze(0)
                          * cos_terms * sin_sq_terms)
            cosm = cosm_terms.sum(1)
            k = self.find_k(cos_target)
            
            ls_target = norm_target_prod * (((-1)**k * cosm) - 2*k)
            logit3[range(batch_size), target] = ls_target
            return logit
        else:
            assert target is None
            return input.matmul(self.weight)

class sphere20a(nn.Module):
    def __init__(self,classnum=3,feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = LSoftmaxLinear(512,self.classnum, 4)

    def forward(self, x, y):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = (self.fc5(x))
#         print(x)
        if self.feature: return x

        x = self.fc6(x)
#         x = self.fc6(x, None)

        return x

#---------------------------------------------------------------------------
# MODEL 1
# global: DenseNet161 (densenet_emotiw_pretrainemotic_lr001)
#---------------------------------------------------------------------------

model1 = models.densenet161(pretrained=False)
num_ftrs = model1.classifier.in_features
model1.classifier = nn.Linear(num_ftrs, 3)

model1 = model1.to(device)
model1 = nn.DataParallel(model1)
model1.load_state_dict(torch.load('../TrainedModels/TrainDataset_my_models/model_1_2_densenet_emotiw_pretrainemotic_lr001.pt', map_location=lambda storage, loc: storage))
model1 = model1.module

print('Pretrained EmotiC DenseNet Loaded! (Model 1)')

#---------------------------------------------------------------------------
# MODEL 2
#  Global resnet18 (resnet18_emotiw_pretrainemotic_lr001.pt)
#---------------------------------------------------------------------------

model2 = torch.load('../TrainedModels/TrainDataset_my_models/model_2_2_resnet18_EmotiW', map_location=lambda storage, loc: storage)

print('Global resnet18 Loaded! (Model 2)')

#---------------------------------------------------------------------------
# MODEL 3
# skeletons Densenet161 (DenseNet161_skeletons_model1)
#---------------------------------------------------------------------------

model3 = torch.load('../TrainedModels/TrainDataset_my_models/model_3_1_DenseNet161_skeletons_model1', map_location=lambda storage, loc: storage)

print('skeletons Densenet161 Loaded! (Model 3)')

#---------------------------------------------------------------------------
# MODEL 4
# skeletons EfficientNet (EfficientNet_skeletons)
#---------------------------------------------------------------------------

model4 = torch.load('../TrainedModels/TrainDataset_my_models/EfficientNet_skeletons_new', map_location=lambda storage, loc: storage)

print('skeletons EfficientNet Loaded! (Model 4)')

#---------------------------------------------------------------------------
# MODEL 5
# crossvit+densenet161  Model (FaceAttention_AlignedModel_FullTrain_lr001_dropout_BN_SoftmaxLr01)
#---------------------------------------------------------------------------

def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d
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

    def forward(self, face_features): #face_features:  32 16 3 96 112

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

    def forward(self, img2): #img2:32 3 256 256
        # x = self.to_patch_embedding(img2)  #x:32 1 256
        x = img2
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens = cls_tokens.to(device)

        x = torch.cat((cls_tokens, x), dim=1)

        pos_embedding = self.pos_embedding
        pos_embedding = pos_embedding.to(device)
        x += pos_embedding[:, :(n + 1)]
        #x += self.pos_embedding[:, :(n + 1)]    # 加position embedding

        return self.dropout(x)
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size = 256,
        num_classes = 256,
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

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))
        self.linear = nn.Linear(256, 3)  #后加的，最后的linear层

    def forward(self, face_features,img2):      #img1:32 3 256 256   img2:32 3 256 256
        sm_tokens = self.sm_image_embedder(face_features)   #face_features：32 16 3 96 112
        lg_tokens = self.lg_image_embedder(img2)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)   #32 17 256    32 2 256

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))  #sm_cls:32 256   lg_cls:32 256

        self.sm_mlp_head =self.sm_mlp_head.to(device)
        self.lg_mlp_head =self.lg_mlp_head.to(device)

        sm_logits = self.sm_mlp_head(sm_cls)   #多头注意  32 256
        lg_logits = self.lg_mlp_head(lg_cls)   #多头注意  32 256
        x = sm_logits + lg_logits  #后加的   x: 32 256
        # x = (self.linear(x))       #后加的，最后的3分类
        return x

crossvit = CrossViT(image_size = 256,num_classes = 256,depth = 4, sm_dim = 256, sm_patch_size = 64, sm_enc_depth = 2,  sm_enc_heads = 8, sm_enc_mlp_dim = 2048,
    lg_dim = 256, lg_patch_size = 256,  lg_enc_depth = 3,  lg_enc_heads = 8, lg_enc_mlp_dim = 2048,   cross_attn_depth = 2,  cross_attn_heads = 8,
    dropout = 0.1, emb_dropout = 0.1)

crossvit = crossvit.to(device)
crossvit = torch.nn.DataParallel(crossvit)

crossvit_save_model = torch.load("../TrainedModels/pre_crossvit_emotiw.pt")
crossvit_model_dict =  crossvit.state_dict()
state_dict = {k:v for k,v in crossvit_save_model.items() if k in crossvit_model_dict.keys()}

crossvit_model_dict.update(state_dict)
crossvit.load_state_dict(crossvit_model_dict)

class FaceAttention(nn.Module):
    def __init__(self, global_model, align_model):
        super(FaceAttention, self).__init__()

        self.global_model = global_model
        self.align_model = align_model

        self.global_fc_main = nn.Linear(2208, 256)
        nn.init.kaiming_normal_(self.global_fc_main.weight)
        self.global_fc_main.bias.data.fill_(0.01)

        self.global_fc3_debug = nn.Linear(512, 3)
        nn.init.kaiming_normal_(self.global_fc3_debug.weight)
        self.global_fc3_debug.bias.data.fill_(0.01)

        self.global_fc_main_dropout = nn.Dropout(p=0.5)
        self.align_model_dropout = nn.Dropout(p=0.5)

        self.bn_debug_face = nn.BatchNorm1d(256, affine=False)
        self.bn_debug_global = nn.BatchNorm1d(256, affine=False)

    def forward(self, image, face_features_initial, numberFaces, labels):
        # image: 32 3 224 224 batchsize  dim 图片大小     face_features_initial:32 15 3 96 112     numberFaces:32
        features = self.global_model.forward(image)  # features: 32 2208 7 7

        out = F.relu(features, inplace=False)  # out: 32 2208 7 7
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0),
                                                                                  -1)  # global_features_initial: 32 2208

        global_features_initial = Variable(global_features_initial)

        global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208

        global_features = self.global_fc_main_dropout(
            self.global_fc_main(global_features_initial))  # global_features:32 256

        global_features = global_features.view(-1, 1, 256)  # global_features:32 1 256

        batch_size = global_features.shape[0]  # batch_size:32

        numberFaces = numberFaces.data.cpu().numpy()  # 改动

        maxNumber = np.minimum(numberFaces, maxFaces)

        face_features = torch.zeros((batch_size, maxFaces, 256), dtype=torch.float)  # face_features:32 15 256   全是零

        face_features = face_features.to(device)

        for j in range(batch_size):
            face = face_features_initial[j]  # face:15 3 96 112
            face_features[j, :, :] = self.align_model.forward(face, labels)  # labels: 32

        face_features = self.align_model_dropout(face_features)
        # ———————————————————————————————————————————————————————————————————————————————————————————————————————————
        # 输入： face_features:32 16 256   global_features:32 1 256
        pred = crossvit(face_features, global_features)  # crossvit 入口     输出：pred:32 256

        # ———————————————————————————————————————————————————————————————————————————————————————————————————————————

        global_features = global_features.view(batch_size, -1)  # 32 256

        pred = self.bn_debug_face(pred)
        global_features = self.bn_debug_global(global_features)
        final_features = torch.cat((pred, global_features), dim=1)  # final_features：32 512

        x = (self.global_fc3_debug(final_features))  # x: 32 3
        return x

path = '../TrainedModels/TrainDataset_my_models/FaceAttention_AlignedModel_FullTrain_lr001_dropout_BN_SoftmaxLr01'
model5 = torch.load(path, map_location=lambda storage, loc: storage)
print('crossvit+densenet161 Model Loaded! (Model 5)')

#---------------------------------------------------------------------------
# ENSEMBLE
#---------------------------------------------------------------------------

class Ensemble(nn.Module):
    def __init__(self, model_1, model_2, model_3, model_4, model_5):
        super(Ensemble, self).__init__()
        
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self.model_4 = model_4
        self.model_5 = model_5

    def forward(self, image, skeleton_image, labels, face_features_mtcnn, face_features_aligned, numberFaces, phase):
        
        output1 = self.model_1(image)
        output2 = self.model_2(image)
        output3 = self.model_3(skeleton_image)
        output4 = self.model_4(skeleton_image)
        output5 = self.model_5(image, face_features_aligned, numberFaces, labels)

        # output = 0* output1 + 5 * output2 + 10 * output3 + 10 * output4 + 1 * output5
        # output = 7.4 * output1 + 7 * output2 + 6.7 * output3 + 6 * output4 + 7.6 * output5
        # output = 0.21 * output1 + 0.2 * output2 + 0.18 * output3 + 0.17 * output4 + 0.22 * output5
        output = 0.12 * output1 + 0.08 * output2 + 0.12 * output3 + 0.08 * output4 + 0.6 * output5

        return output, output1, output2, output3, output4, output5

model_ft = Ensemble(model1, model2, model3, model4, model5)
model_ft = model_ft.to(device)
model_ft = nn.DataParallel(model_ft)
print("Ensemble Loaded.")

#---------------------------------------------------------------------------
# TRAINING
#---------------------------------------------------------------------------

output_train_model1 = []
output_train_model2 = []
output_train_model3 = []
output_train_model4 = []
output_train_model5 = []
output_train = []
output_valid_model1 = []
output_valid_model2 = []
output_valid_model3 = []
output_valid_model4 = []
output_valid_model5 = []
output_valid = []
output_test_model1 = []
output_test_model2 = []
output_test_model3 = []
output_test_model4 = []
output_test_model5 = []
output_test = []
label_train = []
label_valid = []
label_test = []

def train_model(model, criterion, optimizer=None, scheduler=None, num_epochs = 1):
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in range(0, 3):            
            if phase == 0:
                dataloaders = train_dataloader
                # scheduler.step()
                model.eval()
            elif phase == 1:
                dataloaders = val_dataloader
                model.eval()
            elif phase == 2:
                dataloaders = test_dataloader
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for i_batch, sample_batched in enumerate(dataloaders):
                inputs = sample_batched['image']
                skeleton_image = sample_batched['skeleton_image']
                labels = sample_batched['label']
                face_features_mtcnn = sample_batched['features_mtcnn']
                face_features_aligned = sample_batched['features_aligned']
                numberFaces = sample_batched['numberFaces']

                inputs = inputs.to(device)
                skeleton_image = skeleton_image.to(device)
                labels = labels.to(device)
                face_features_mtcnn= face_features_mtcnn.to(device)
                face_features_aligned = face_features_aligned.to(device)
                numberFaces = numberFaces.to(device)
                                
                with torch.set_grad_enabled(phase == 0):
                    outputs, output1, output2, output3, output4, output5 = model(inputs, skeleton_image, labels, face_features_mtcnn, face_features_aligned, numberFaces, phase)

                    if phase == 0:
                        output_train.extend(outputs.data.cpu().numpy())
                        output_train_model1.extend(output1.data.cpu().numpy())
                        output_train_model2.extend(output2.data.cpu().numpy())
                        output_train_model3.extend(output3.data.cpu().numpy())
                        output_train_model4.extend(output4.data.cpu().numpy())
                        output_train_model5.extend(output5.data.cpu().numpy())
                        label_train.extend(labels.cpu().numpy())
                    elif phase == 1:
                        output_valid.extend(outputs.data.cpu().numpy())
                        output_valid_model1.extend(output1.data.cpu().numpy())
                        output_valid_model2.extend(output2.data.cpu().numpy())
                        output_valid_model3.extend(output3.data.cpu().numpy())
                        output_valid_model4.extend(output4.data.cpu().numpy())
                        output_valid_model5.extend(output5.data.cpu().numpy())
                        label_valid.extend(labels.cpu().numpy())
                    else:
                        output_test.extend(outputs.data.cpu().numpy())
                        output_test_model1.extend(output1.data.cpu().numpy())
                        output_test_model2.extend(output2.data.cpu().numpy())
                        output_test_model3.extend(output3.data.cpu().numpy())
                        output_test_model4.extend(output4.data.cpu().numpy())
                        output_test_model5.extend(output5.data.cpu().numpy())
                        label_test.extend(labels.cpu().numpy())

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            np.savez('fourteen_models_output_data_shiyan',
            output_train = output_train,
            output_train_model1 = output_train_model1,
            output_train_model2 = output_train_model2,
            output_train_model3 = output_train_model3,
            output_train_model4 = output_train_model4,
            output_train_model5 = output_train_model5,
            output_valid = output_valid,
            output_valid_model1 = output_valid_model1,
            output_valid_model2 = output_valid_model2,
            output_valid_model3 = output_valid_model3,
            output_valid_model4 = output_valid_model4,
            output_valid_model5 = output_valid_model5,
            output_test = output_test,
            output_test_model1 = output_test_model1,
            output_test_model2 = output_test_model2,
            output_test_model3 = output_test_model3,
            output_test_model4 = output_test_model4,
            output_test_model5 = output_test_model5,
            label_train = label_train, 
            label_valid = label_valid, 
            label_test = label_test)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    time_elapsed = time.time() - since
    print('Training complete in {: .0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    return model

criterion = nn.CrossEntropyLoss()

model = train_model(model_ft, criterion, None, None, num_epochs=epochs)

np.savez('fourteen_models_output_data_shiyan',
output_train = output_train,
output_train_model1 = output_train_model1,
output_train_model2 = output_train_model2,
output_train_model3 = output_train_model3,
output_train_model4 = output_train_model4,
output_train_model5 = output_train_model5,
output_valid = output_valid,
output_valid_model1 = output_valid_model1,
output_valid_model2 = output_valid_model2,
output_valid_model3 = output_valid_model3,
output_valid_model4 = output_valid_model4,
output_valid_model5 = output_valid_model5,
output_test = output_test,
output_test_model1 = output_test_model1,
output_test_model2 = output_test_model2,
output_test_model3 = output_test_model3,
output_test_model4 = output_test_model4,
output_test_model5 = output_test_model5,
label_train = label_train, 
label_valid = label_valid, 
label_test = label_test)