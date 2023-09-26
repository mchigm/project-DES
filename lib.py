import cv2
import statistics
import pywt
import torch, torchgen
import json
import pandas as pd
import termcolor as color
import random as rd
import numpy as np
import xgboost as xgb
from sklearn import base, compose, covariance, ensemble, exceptions, experimental, feature_extraction, gaussian_process, kernel_approximation, kernel_ridge, linear_model, neural_network, pipeline, preprocessing, tree, metrics
from matplotlib import pyplot as plt
from numbers import Number as n
from scipy import cluster, constants, fftpack, integrate, interpolate, io, linalg, ndimage, optimize, sparse, spatial, stats
from urllib import error, request
from bs4 import BeautifulSoup as bf
from urllib3 import connection, response
from torch import nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class dnn:
    #Setup Net
    class Net(nn.Module):
        def __init__(self):
            super(dnn.Net, self).__init__()

            # 2D convolutional layer from image
            self.conv1 = nn.Conv2d(1, 32, 3, 1) #32 features, kernel size 3
            # 2D convolutional layer from 32 feartures
            self.conv2 = nn.Conv2d(32, 54, 3, 1) #64 features, kernel size 3

            # Ensurance
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)

            # First layer
            self.fc1 = nn.Linear(9216, 128)
            # Second layer
            self.fc2 = nn.Linear(128, 10)# 10 labels
        
        def forward(self, data):
            # Data in conv1
            data = self.conv1(data)

            # [Reflected Linear] activation
            data = F.relu(data)

            data = self.conv2(data)

            data = F.relu(data)

            # Max over data
            data = F.max_pool2d(data, 2)
            data = self.dropout1(data)
            data = torch.flatten(data, 1)
            data = self.fc1(data)
            data = F.relu(data)
            data = self.dropout2(data)
            data = self.fc2(data)

            # Softmax
            o = F.log_softmax(data, dim = 1)
            return o

    my_nn = Net()
    print(my_nn)

class cnn:
    m = resnet50()
    train_nodes, eval_nodes = get_graph_node_names(resnet50)

    return_nodes = {
        # node_name: user-specified key for output dict
        'layer1.2.relu_2': 'layer1',
        'layer2.3.relu_2': 'layer2',
        'layer3.5.relu_2': 'layer3',
        'layer4.2.relu_2': 'layer4',
    }

    return_nodes = {
        'layer1': 'layer1',
        'layer2': 'layer2',
        'layer3': 'layer3',
        'layer4': 'layer4',
    }

    create_feature_extractor(m, return_nodes=return_nodes)

    class Resnet50WithFPN(torch.nn.Module):
        def __init__(self):
            super(cnn.Resnet50WithFPN, self).__init__()
            m = resnet50()
            self.body = create_feature_extractor(
                m, return_nodes={f'layer{k}': str(v)
                                for v, k in enumerate([1, 2, 3, 4])})
            inp = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                out = self.body(inp)
            in_channels_list = [o.shape[1] for o in out.values()]
            self.out_channels = 256
            self.fpn = FeaturePyramidNetwork(
                in_channels_list, out_channels=self.out_channels,
                extra_blocks=LastLevelMaxPool())

        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)
            return x


    model = MaskRCNN(Resnet50WithFPN(), num_classes=91).eval()

class extractFeature:
    def 时域特征(value):
        # Define function variables
        #value
        data = value #create copy

        # 有纲量
        max = np.amax(value, axis = 1) #maximum
        ap = np.amax(abs(value), axis = 1) #absolute maximum
        min = np.amin(value, axis = 1) #minimum
        m = np.amin(value, axis = 1) #mean
        pp = np.amax(value, axis = 1) - np.amin(value, axis = 1) #peak-peak-value
        am = np.mean(abs(value), axis = 1) #absolute mean
        ms = np.sqrt(np.sum(value**2, axis = 1) / value.shape) #mean square root
        sa = (np.sum(np.sqrt(abs(value)), axis = 1) / value.shape) ** 2 #square root amplitude
        vars = np.var(value, axis = 1) #variance
        sd = np.std(value, axis = 1) #standard deviance
        ks = stats.kurtosis(value, axis = 1) #kurtosis
        skew = stats.skew(value, axis = 1) #skewness
        ma = np.sum(np.abs(value), axis = 1) / value.shape #mean amplitude

        # 无纲量
        cl_f = np.amax(abs(value), axis = 1) / ((np.sum(np.sqrt(abs(value)), axis = 1) / value.shape) ** 2) #clearance
        s_f = np.sqrt(np.sum(value**2, axis = 1) / value.shape) / np.amax(abs(value), axis = 1) #shape
        i_f = np.amax(abs(value), axis = 1) / np.amax(abs(value), axis = 1) #impulse
        cr_f = np.amax(abs(value), axis = 1) / np.sqrt(np.sum(value**2, axis = 1) / value.shape) #crest
        k_f = stats.kurtosis(value, axis = 1) / (np.sqrt(np.sum(value**2, axis = 1) / value.shape) ** 4) #kurtosis

        return np.array([max, ap, min, m, pp, am, ms, sa, vars, sd, ks, skew, ma, cl_f, s_f, i_f, cr_f, k_f], data).T
    
    def 频域特征(value, freq):
        # Define function variables
        # Shape --> (m,n), a 2-D array
        vf = np.fft.fft(value, axis = 1) 
        m, n = np.fft.fft(value, axis = 1).shape # m=sample number, n = signal length

        # Bernoulli transformation
        sva = np.abs(vf)[:,:n//2] #signal amplitude 1
        fa = np.fft.fftfreq(n, 1/freq)[:n//2] #frequency 1
        svb = np.abs(vf)[:,n//2:] #signal amplitude 2
        #fb = facebook #:d
        fb = np.fft.fftfreq(n, 1/freq)[n//2:] #frequency 2
        sv = np.mean([sva, svb], axis = 1) #overall amplitude
        f = np.mean([fa, fb], axis = 1) #overall frequency
        ps = sv ** 2 / n #power specturm

        #无纲量
        c = np.sum(f * ps, axis = 1) / np.sum(ps, axis = 1) #centeroid
        mf = np.mean(ps, axis = 1) #frequency mean
        msf = np.sqrt(np.sum(ps * np.square(f), axis = 1) / np.sum(ps, axis = 1)) #mean square root frequency
        f_t = np.tile(f.reshape(1, -1), (m, 1)) #frequency row
        c_t = np.tile(c.reshape(-1, 1), (1,f_t.shape[1])) #frequency lane
        fv = np.sum(np.square(f_t - c_t) * ps, axis = 1) / np.sum(ps, axis = 1) #frequency variance

        return np.array([c, mf, msf, fv], [vf, sv, f, ps, f_t, c_t], value, freq).T
    
    def 时频特征(value, wavelet = "db3", mode = "symmetric", maxlevel = 3):
        result = []
        for i in range(value.shape[0]):
            wpf = f(value[i])
            result.append(wpf)
        result = np.array(result)
        feature = result.shape
        def f(x):
            #小波泡特征
            # Shape --> (n, ), a 1-D array
            wp = pywt.WaveletPacket(x, wavelet = wavelet, mode = mode, maxlevel = maxlevel)

            ns = [node.path for node in wp.get_level(maxlevel, "natural")] #final node path

            e = [] #node energy
            for node in ns:
                ei = np.linalg.norm(wp[node].data, ord = None) ** 2 #segment energy # Get norm, sqrt(norm) = segment energy
                e.append(ei)
            
            # e = vector(feature)

            #E% == vector(feature), E% > [0, 100)
            te = np.sum(e) #E_total
            f = [] #fetures
            for ei in e:
                f.append(ei / te * 100) #E$
            
            return np.array(f)
        
        return feature