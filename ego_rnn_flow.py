import torch
import resnet_split
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from flow_layer import FlowLayer

class FlowAugmentedAttention(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(FlowAugmentedAttention, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnet_split.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.flow_layer = FlowLayer(channels=128)
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable, repr_flow=True, no_cam=False):
        if repr_flow:
            state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                     Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
            mid_features = []
            flows = []
            for t in range(inputVariable.size(0)):
                conv3 = self.resNet(inputVariable[t], split=0)
                mid_features.append(conv3)
            mid_features = torch.stack(mid_features).permute(1,2,0,3,4)
            estimated_flow = self.flow_layer(mid_features)
            flows = estimated_flow.permute(2,0,1,3,4)
            for t in range(flows.size(0)):
                logit, feature_conv, feature_convNBN = self.resNet(flows[t], split=1)
                if not no_cam:    
                    bz, nc, h, w = feature_conv.size()
                    feature_conv1 = feature_conv.view(bz, nc, h*w)
                    probs, idxs = logit.sort(1, True)
                    class_idx = idxs[:, 0]
                    cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
                    attentionMAP = F.softmax(cam.squeeze(1), dim=1)
                    attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
                    attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
                    state = self.lstm_cell(attentionFeat, state)
                else:
                    state = self.lstm_cell(feature_convNBN, state)
            feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
            feats = self.classifier(feats1)
            return feats, feats1
        else:
            state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                     Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
            for t in range(inputVariable.size(0)):
                conv3 = self.resNet(inputVariable[t], split=0)
                logit, feature_conv, feature_convNBN = self.resNet(conv3, split=1)
                bz, nc, h, w = feature_conv.size()
                feature_conv1 = feature_conv.view(bz, nc, h*w)
                probs, idxs = logit.sort(1, True)
                class_idx = idxs[:, 0]
                cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
                attentionMAP = F.softmax(cam.squeeze(1), dim=1)
                attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
                attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
                state = self.lstm_cell(attentionFeat, state)
            feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
            feats = self.classifier(feats1)
            return feats, feats1