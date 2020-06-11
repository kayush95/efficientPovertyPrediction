import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as torchinit
import math
from torch.nn import init, Parameter
import copy
import pdb
import torch

from models import base
import utils

class FlatResNet(nn.Module):

    def seed(self, x):
        raise NotImplementedError

    def forward(self, x):
        x = self.seed(x)
        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                x = F.relu(residual + self.blocks[segment][b](x))
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_single(self, x):
        x = self.seed(x)
        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
           for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                x = residual + self.blocks[segment][b](x)
                x = F.relu(x)
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_full(self, x):
        x = self.seed(x)

        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                x = F.relu(residual + self.blocks[segment][b](x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FlatResNet32(FlatResNet):

    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = base.conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = base.DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample

# Regular Flattened Resnet, tailored for Imagenet etc.
class FlatResNet224(FlatResNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(FlatResNet224, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        strides = [1, 2, 2, 2]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.layer_config = layers

    def seed(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers, downsample

class Policy32(nn.Module):

    def __init__(self, layer_config=[1,1,1], num_blocks=15):
        super(Policy32, self).__init__()
        self.features = FlatResNet32(base.BasicBlock, layer_config, num_classes=10)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()

        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, num_blocks)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy32, self).load_state_dict(state_dict)

    def forward(self, x):
        x = self.features.forward_full(x)
        value = self.vnet(x)
        probs = F.sigmoid(self.logit(x))
        return probs, value

class Policy224(nn.Module):

    def __init__(self, layer_config=[1,1,1,1], num_blocks=289, num_feat=10):
        super(Policy224, self).__init__()
        self.features = FlatResNet224(base.BasicBlock, layer_config, num_classes=1000)
        self.features.avgpool = nn.AvgPool2d(4)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        # print('feat_dim', self.feat_dim)
        # print('features.fc', self.features.fc)
        self.features.fc = nn.Sequential()
        # print('features.fc', self.features.fc)
        self.tile_feat = nn.Linear(self.feat_dim, num_feat)
        # print('tile_feat', self.tile_feat)
        self.logit = nn.Linear(num_blocks*num_feat, num_blocks)
        self.vnet = nn.Linear(num_blocks*num_feat, 1)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy224, self).load_state_dict(state_dict)

    def forward(self, x):
        tile_all = []
        # print('x.shape', x.shape)
        for tile_idx in range(x.shape[1]):
            x1 = x[:, tile_idx, :, :, :]
            x1 = F.avg_pool2d(x1, 2)
            x1 = self.features.forward_full(x1)
            x1 = self.tile_feat(x1)
            tile_all.append(x1)

        tile_all = torch.stack(tile_all, dim=1)
        # print('tile_all.shape', tile_all.shape)
        tile_all = torch.flatten(tile_all, start_dim=1, end_dim=-1)
        # print('tile_all.shape', tile_all.shape)

        value = self.vnet(tile_all)
        probs = torch.sigmoid(self.logit(tile_all))      
        return probs, value

class Policy224GRU(nn.Module):

    def __init__(self, layer_config=[1,1,1,1], num_blocks=289, num_feat=128):
        super(Policy224GRU, self).__init__()
        self.features = FlatResNet224(base.BasicBlock, layer_config, num_classes=1000)
        self.features.avgpool = nn.AvgPool2d(4)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        # print('feat_dim', self.feat_dim)
        # print('features.fc', self.features.fc)
        self.features.fc = nn.Sequential()
        # print('features.fc', self.features.fc)
        self.tile_feat = nn.Linear(self.feat_dim, num_feat)
        # print('tile_feat', self.tile_feat)
        self.logit = nn.Linear(num_blocks*num_feat, num_blocks)
        self.vnet = nn.Linear(num_blocks*num_feat, 1)


        self.input_dim = 128
        self.output_dim = 1
        # self.drop_prob = 0.2
        self.hidden_dim = 64
        self.n_layers = 1
        # self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.drop_prob)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()


    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy224GRU, self).load_state_dict(state_dict)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, int(batch_size/4), self.hidden_dim).cuda()
        # hidden = torch.zeros(batch_size, self.n_layers, self.hidden_dim).cuda()
        # hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        # hidden = torch.zeros(self.n_layers, 1, self.hidden_dim).cuda()
        return hidden

    # def forward(self, x, h):
    def forward(self, x):
        tile_all = []
        # print('x.shape ascs', x.shape)
        for tile_idx in range(x.shape[1]):
            x1 = x[:, tile_idx, :, :, :]
            x1 = F.avg_pool2d(x1, 2)
            x1 = self.features.forward_full(x1)
            x1 = self.tile_feat(x1)
            tile_all.append(x1)

        # print('tile_all.shape', tile_all.shape)
        tile_all = torch.stack(tile_all, dim=1)
        # print('tile_all.shape', tile_all.shape)
        self.gru.flatten_parameters()
        # out, h = self.gru(tile_all, h)
        out, h = self.gru(tile_all)
        # print('out.shape', out.shape)
        # print('h', h.shape)
        out = self.fc(self.relu(out))
        # print('out.shape', out.shape)
        probs = torch.sigmoid(out).squeeze(2)  
        # print('probs.shape', probs.shape)  
        value = None  
        return probs, value


class PolicySeq(nn.Module):
    def __init__(self, num_output=1, gamma=1):
        super(PolicySeq, self).__init__()

        self.features = torchmodels.resnet18(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False

        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Linear(num_ftrs, num_output)
        
        # self.gamma = gamma
        
        # # Episode policy and reward history 
        # self.policy_history = Variable(torch.Tensor()) 
        # self.reward_episode = []
        # # Overall reward and loss history
        # self.reward_history = []
        # self.loss_history = []

    def forward(self, x):    
        # model = torch.nn.Sequential(
        #     self.features,
        #     nn.Softmax(dim=-1)
        # )
        x = self.features(x)
        probs = torch.sigmoid(x)
        return probs

class Policy2x2(nn.Module):

    def __init__(self, layer_config=[1,1,1,1], num_blocks=4):
        super(Policy2x2, self).__init__()
        self.features = FlatResNet224(base.BasicBlock, layer_config, num_classes=1000)
        self.features.avgpool = nn.AvgPool2d(4)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()

        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, 1)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy2x2, self).load_state_dict(state_dict)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = self.features.forward_full(x)
        value = self.vnet(x)
        probs = torch.sigmoid(self.logit(x))
        return probs

class Scale(nn.Module):
   def __init__(self):
       super(Scale, self).__init__()
       self.scale = nn.Parameter(torch.FloatTensor(torch.rand(10)))

   def forward(self, input):
       return input * self.scale