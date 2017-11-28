
import progressbar
import sys
import argparse
import torch
from copy import deepcopy
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.utils.data as data

from tensorboard_logger import configure, log_value, Logger

import itertools
from itertools import izip
from matplotlib import  pyplot as plt
import os,os.path
import glob
from time import time
import shutil
import hickle as pickle
from PIL import Image
import collections
import math

batch_size = 128
base_lr = .1
lr_drop_freq=10
criterion = nn.CrossEntropyLoss()
num_workers = 0
from os.path import expanduser
homeDir = expanduser('~')
sys.path.append(os.path.join(homeDir,'YellowFin_Pytorch/tuner_utils/')) # yellowfin :-)
from yellowfin import YFOptimizer

def matVar(size=(1,3,64,64),cuda=False):
    v = Variable(torch.randn(size))
    if cuda:
        v = v.cuda()
    return v

#def adjust_learning_rate(optimizer, epoch, base_lr, lr_drop_freq, gamma=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
#    lr = base_lr * (gamma ** (epoch // lr_drop_freq))
#    for param_group in optimizer.param_groups:
#

def adjust_learning_rate(optimizer, epoch, base_lr, lr_drop_freq = 100, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by gamma every K epochs"""                
    if (epoch + 1) % lr_drop_freq == 0: # Note this works only for continuous mode (not stopping+loading)
        if type(optimizer) is YFOptimizer:
            optimizer.set_lr_factor(optimizer.get_lr_factor() * gamma)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * gamma

def train(model,epoch,optimizer,maxIters=np.inf,targetTranslator=None,train_loader=None, criterion=None,criterion2 = None,disableBatchNorm=False,cuda=True, balancing_factor = 0.0, logger=None):
    T0 = time()
    if not disableBatchNorm:
        model.train()
    else:
        model.eval()
    
    nBatches = 0    
    running_loss = 0.0
    running_loss2 = 0.0
    losses = []
    nSamples=0
    maxIters = min(maxIters,len(train_loader))
    startTime = time()
    for batch_idx, (data, target) in enumerate(train_loader):     
        
        target = target.long().squeeze()
        if targetTranslator is not None:            
            target2 = targetTranslator(target.clone())
            target2 = data.cuda(), targfet.cuda(),target2.cuda()
        target = target.long().squeeze()
        if cuda:
            data, target = data.cuda(), target.cuda()                        
        data, target = Variable(data), Variable(target)
        #, Variable(target2)
        optimizer.zero_grad()
        
        output = model(data)
        if type(output) is tuple:
            gates = output[1]
            output = output[0]
                
        #output = model(data)
                        
        loss = criterion(output, target)# + criterion(output2,target2)        
        if criterion2 is not None and balancing_factor > 0:
            loss2 = criterion2(gates)
            loss+= balancing_factor * loss2
        else:
            loss2 = 0
                    
        #        

        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
        running_loss += loss.data[0]
        if criterion2 is not None and balancing_factor > 0:
            running_loss2 += loss2.data[0]/balancing_factor
        else:
            running_loss2 = -1
        
        
    
        
        nBatches+=1#len(data)
        nSamples+=len(data)
        if batch_idx % 5 == 0 and time()-T0 > .1:
            T0 = time()
            elapsedTime = time()-startTime
            S = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}\tAvg Loss 2: {:.6f} ({:.2f} imgs/sec)'.format(epoch, batch_idx * len(data),
                                                                             len(train_loader.dataset), 
                                                                                 100. * batch_idx / len(train_loader), 
                                                                                 running_loss/nBatches,running_loss2/(nBatches),
                                                                                 nSamples/elapsedTime)
            if logger is not None:
                logger.log_value('training loss',loss.data[0],batch_idx + epoch * maxIters)
            print '\r{}'.format(S),
        if batch_idx > maxIters:
            break
    #b1
    if logger is not None:
        if hasattr(optimizer,'param_groups'):
            
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                logger.log_value('learning rate',cur_lr,epoch)
            
        if hasattr(optimizer,'get_lr_factor'):
            logger.log_value('learning rate',optimizer.get_lr_factor(),epoch)
            
    return losses
            
def test(model,epoch,targetTranslator=None,test_loader=None,prev_acc=0,alpha=None,criterion=None, maxIters=np.inf,cuda=True, logger=None):
    assert (criterion is not None)
    #criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    nSamples = 0
    maxIters = min(maxIters,len(test_loader))
    for batch_idx, (data, target) in enumerate(test_loader):
        target = target.long().squeeze()
        if targetTranslator is not None:            
            target2 = targetTranslator(target.clone())
            target2 = target2.cuda()
            Variable(target2)
        if cuda:
            data, target = data.cuda(), target.cuda()
        
        data, target = Variable(data), Variable(target)
        if alpha is not None:            
            output = model(data,alpha)
        else:
            #b1
            output = model(data)
            if type(output) is tuple:
                gates = output[1]
                output = output[0]
        cur_test_loss = criterion(output, target).data[0]
        test_loss += cur_test_loss 
                        
        pred = output.data.max(1)[1] # get the index of the max log-probability        
        correct += pred.eq(target.data).cpu().sum()
        nSamples+=len(data)
        if batch_idx >= maxIters:
            break

    test_loss /= len(test_loader) # loss function already averages over batch size
    if logger is not None:
            logger.log_value('test loss',test_loss,epoch)
    cur_acc = 100. * correct / nSamples
    #if prev_acc < cur_acc:
    P = '({}) :Test set: Avg. loss: {:.4f}, Acc: {}/{} ({:.1f}%)'.format(epoch,
    test_loss, correct, nSamples, cur_acc)
    if logger is not None:        
        logger.log_value('test accuracy',cur_acc,epoch)
        
    print '\r{}'.format(P),
    return 100. * correct / nSamples

def checkModelConsistency(newModel,oldModel):
    for a_fine,a_orig in zip(newModel, oldModel):
        tt = type(a_orig)    
        if tt is nn.Conv2d:    
            print '*',
            w_fine = a_fine.w.transpose(0,1).contiguous().view(a_fine.s)
            w_orig = a_orig.weight    
            assert( (w_fine-w_orig).data.sum() ==0)
#checkModelConsistency(f_fine_m,model_10.features.children())    


def save_checkpoint(state, is_best, epoch, modelDir):
    """Saves checkpoint to disk"""
    checkPointPath = '{}/{}'.format(modelDir,str(epoch).zfill(4))
    torch.save(state, checkPointPath)
    if is_best:
        shutil.copyfile(checkPointPath, '{}/{}'.format(modelDir,'best'))
        
def defaultCallBacks():
    return {'trainEpochStart':[],'trainEpochEnd':[],'testEpochStart':[],'testEpochEnd':[]}

def trainAndTest(model,optimizer=None,modelDir=None,epochs=5,targetTranslator=None,model_save_freq=20,
                train_loader=None,test_loader=None,stopIfPerfect=True,  criterion=nn.CrossEntropyLoss(),
                criterion2 = None, adjust_learning_rate=adjust_learning_rate, maxIters=np.inf,base_lr=base_lr,
                lr_drop_freq=lr_drop_freq,disableBatchNorm=False,cuda=True,balancing_factor=0.0,logger=None,
                callbacks=defaultCallBacks(),gamma=.1):
             
    last_epoch = 0
    corrects = []
    
    needToSave = modelDir is not None and model_save_freq > 0
    all_accuracies = []
    if needToSave:
    
        if not os.path.isdir(modelDir):
            os.makedirs(modelDir)    

        
        g = list(sorted(glob.glob(os.path.join(modelDir,'*'))))
        g = [g_ for g_ in g if not 'best' in g_]
        
        g_new = []
        for gg in g: # fixing file names to be zero padded    
            g1,g2 = os.path.split(gg)
            newName = '/'.join([g1,g2.zfill(4)])
            if gg <> newName:
                print 'moving'
                print gg,'to'
                print newName
                shutil.move(gg,newName)
            g_new.append(newName)
        g = list(sorted(g_new))
        
        if len(g) > 0:
            lastCheckpoint = g[-1]
            # load the last checkpoint
            print 'loading from', lastCheckpoint
            
            checkpoint = torch.load(lastCheckpoint)
            last_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            all_accuracies = checkpoint.get('all_accuracies',all_accuracies)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(lastCheckpoint))
            
    best_acc = 0
    all_losses = []
    hasCallBacks = callbacks is not None
    
    for epoch in range(last_epoch, epochs): # epochs + 1):
        
        if hasCallBacks:
            for callback in callbacks['trainEpochStart']:
                callback(model,optimizer,epoch)
                        
        if adjust_learning_rate is not None:
            adjust_learning_rate(optimizer,epoch,base_lr,lr_drop_freq,gamma)
        losses = train(model=model,epoch=epoch,optimizer=optimizer,targetTranslator=targetTranslator,
              train_loader=train_loader,criterion=criterion,criterion2 = criterion2, maxIters=maxIters,disableBatchNorm=disableBatchNorm,cuda=cuda,
                        balancing_factor=balancing_factor,logger=logger)
        
        if hasCallBacks:
            for callback in callbacks['trainEpochEnd']:
                callback(model,optimizer,epoch)
        
        all_losses.extend(losses)
        print
        if hasCallBacks:
            for callback in callbacks['testEpochStart']:
                callback(model,optimizer,epoch)
        
        cur_acc = test(model,epoch,targetTranslator=targetTranslator,test_loader=test_loader,
                       prev_acc=best_acc,criterion=criterion, maxIters=maxIters,cuda=cuda,logger=logger)        
        if hasCallBacks:
            for callback in callbacks['testEpochEnd']:
                callback(model,optimizer,epoch)
        all_accuracies.append(cur_acc)
        corrects.append(cur_acc)
        print
                    
        
        if needToSave and (epoch % model_save_freq == 0 or epoch == epochs-1):
            print 'saving model...',
            checkPointPath = '{}/{}'.format(modelDir,epoch)
            if cur_acc > best_acc:
                best_acc = cur_acc
                is_best = True
            else:
                is_best = False
            save_checkpoint({
            'epoch': epoch + 1,
            'all_losses':all_losses,
            'all_accuracies':all_accuracies,
            'last_epoch_losses':losses,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'cur_acc': cur_acc
            }, is_best, epoch, modelDir)                                    
        #if cur_acc>=99.5:
        #    break
                
    return corrects


def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = npimg-npimg.min()
    npimg = npimg/npimg.max()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) and m.affine:            
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        #elif isinstance(m, nn.Linear):
        #    init.normal(m.weight, std=1e-3)
        #    if m.bias:
        #        init.constant(m.bias, 0)
                
                

class VGG_backcomp(nn.Module):
    def __init__(self, features, fc_size=512,num_classes=1000,dropout=True,fullyconv=False):
            super(VGG, self).__init__()
            self.features = features
            self.fullyconv = fullyconv
            if not fullyconv:            
                
                if dropout:
                
                    self.classifier = nn.Sequential(
                        nn.Linear(fc_size, 512),
                        nn.ReLU(True),
                    nn.Dropout(),                
                        nn.Linear(512, num_classes),
                    )
                else:
                    self.classifier = nn.Sequential(
                        nn.Linear(fc_size, 512),
                        nn.ReLU(True),                    
                        nn.Linear(512, num_classes),
                    )
            else:
                self.classifier = nn.Sequential(nn.Linear(512,num_classes)) # get just the last layer,yes?
            
    def forward(self, x):
        
        x = self.features(x)
        #print 'x size:',x.size()
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
class VGG(nn.Module):
    def __init__(self, features, fc_size=512,num_classes=1000,dropout=True,fullyconv=False):
        super(VGG, self).__init__()
        self.features = features
        self.fullyconv = fullyconv
        if not fullyconv:            
            
            if dropout:
            
                self.classifier = nn.Sequential(
                    nn.Linear(fc_size, 512),
                    nn.ReLU(True),
                nn.Dropout(),                
                    nn.Linear(512, num_classes),
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(fc_size, 512),
                    nn.ReLU(True),                    
                    nn.Linear(512, num_classes),
                )
        else:
            self.classifier = nn.Sequential(nn.Conv2d(512,num_classes,2,2)) # get just the last layer,Yes?
        init_params(self)
    def forward(self, x):
        
        x = self.features(x)
        #print 'x size:',x.size()
        if not self.fullyconv:
            x = x.view(x.size(0), -1)        
        x = self.classifier(x)
        if self.fullyconv:
            x = x.view(x.size(0), -1)
        return x,None

# In[3]:
#cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class AlphaNet(nn.Module):
    def __init__(self, features, classifier, otherClassifier=None):
            super(AlphaNet, self).__init__()
            if type(features) is list:
                self.features = nn.Sequential(*features)
            else:
                self.features = features            
            self.classifier = classifier
            self.otherClassifier = otherClassifier
            self.outputSize = None
    def getControlParams(self):
        # return parameters of all layers, except convolutional.
        params = []
        for q in self.features:
            q_type = type(q)
            if q_type is nn.Conv2d or q_type is nn.BatchNorm2d:
                continue
            if q_type is controlledConv: # probably nothing else
                params.extend(list(q.parameters()))
                params.append(q.bias)
            
        params.extend(list(self.classifier.parameters()))
        return params
            
    
    def extendToSize(self,x):
        S = self.outputSize        
        if S is not None:
            s = x.size()
            assert s[1] <= S, 'output larger than required output size'
            if s[1] < S:
                XX = Variable(torch.zeros(s[0],S).cuda())
                XX[:,:s[1]] = x
                x = XX
        return x
            
    
    def forward(self, x, alpha=None):       
        for f in self.features:
            if type(f) is controlledConv:
                x = f(x,alpha)
            else:
                x = f(x)
        
        x = x.view(x.size(0), -1)
        
        if alpha is None:        
            x = self.classifier(x)
        else:
            assert self.otherClassifier is not None, 'cannot use alpha without other classifier'
            #assert self.outputSize is not None , 'cannot use alpha without specified output size'
            x1 = self.classifier(x)                                    
            x2 = self.otherClassifier(x)
            
            # set the desired output to the maximum between the two classes
            if self.outputSize is None:
                print 'automatically determining maximal output size...'
                self.outputSize = max(x1.size()[1],x2.size()[1])
            #print 'sizes before:',x1.size(),x2.size()
            x1 = self.extendToSize(x1)
            x2 = self.extendToSize(x2)
            #print 'sizes after:',x1.size(),x2.size()
            
            myAlpha = alpha.expand_as(x1)
            x = myAlpha * x1 + (1-myAlpha) * x2
        return x

def replaceLastLayer(model,num_outputs):
    mod = list(model.children())
    mod.pop()
    mod.append(torch.nn.Linear(512, num_outputs))
    model = torch.nn.Sequential(*mod)
    return model
def freezeBatchNormLayers(model):        
    if hasattr(model,'features'):    
        for p in model.features.children():
            
            if type(p) is nn.BatchNorm2d:
                print '.',
                for q in p.parameters():
                    
                    q.requires_grad = False
        for p in model.classifier.children():            
            if type(p) is nn.BatchNorm2d:
                print '.',
                for q in p.parameters():
                    
                    q.requires_grad = False
    else:
        for p in model.children():
            if type(p) is nn.BatchNorm2d:                
                print '.',
                for q in p.parameters():
                    
                    q.requires_grad = False
def ton(V):
    if type(V) is not Variable:
        return V.cpu().numpy()
    else:
        return V.data.cpu().numpy()
def showmat(M):
    if type(M) is not np.ndarray:
        M = ton(M)
    plt.matshow(M)

def countModelParameters(model,need_require_grad=True):
    return sum([p.data.nelement() for p in model.parameters() if p.requires_grad or not need_require_grad])

normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
cuda=True
kwargs = {'num_workers': num_workers, 'pin_memory': False} 


def quickTest(model,test_loader,alpha=None,maxSamples=100000):

    #criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    nPoints = 0    
    for idx, (data, target) in enumerate(test_loader):
        target = target.long().squeeze()
        nPoints += len(target)        
        data, target = Variable(data.cuda()), Variable(target.cuda())    
        if alpha is not None:
            output = model(data,alpha)
        else:
            output = model(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        if nPoints >= maxSamples:
            break    
    cur_acc = 100. * correct / nPoints
    #if prev_acc < cur_acc:
    P = 'Test set: Acc: {}/{} ({:.1f}%)'.format(correct, nPoints, cur_acc)    
    print '\r{}'.format(P),    
    return cur_acc




    
# Initialize from scratch.

from numpy.linalg import lstsq

class conv2d_bn(nn.Module):
    def __init__(self, conv, bn):
        super(conv2d_bn, self).__init__()
        self.conv = conv
        self.bn = bn
    def forward(self,x):
        return self.bn(self.conv(x))
    

class controlledConv(nn.Module):
    def __init__(self, conv, X = None,bias = None, sparse = False, diagonal=False):
        super(controlledConv, self).__init__()
        self.padding = conv.padding
        self.stride = conv.stride
        self.dilation = conv.dilation
        self.conv = conv        
        # Copy the weights as a constant from the original convolution --
        # just to make sure it doesn't change                
        s = conv.weight.size()
        #print 'size of conv:',s
        self.s = list(s)
        w = Variable(torch.Tensor(s).copy_(conv.weight.data))
        w = w.view(s[0],-1).transpose(0,1)
        #print 'size of flattened weights' ,w.size()
        self.w = w.detach().cuda()
        
        self.my_bn = None
        s = conv.weight.size()
        R = s[0]
        #print 'size of X:',X.size()
        L = nn.Linear(X.size()[1],X.size()[0],bias=False)
        #print 'L:',L
        if X is None:
            L.weight.data = torch.eye(R) # Initilize to unit (e.g, keep configuration)
        else:
            L.weight.data = X
        self.L = L
        self.s[0] = L.weight.size()[0]
	hasBias = bias is not None
        if hasBias:
            s_bias = self.s[0]
            self.conv_bias = Variable(torch.Tensor(conv.bias.data.size()).copy_(conv.bias.data))
            self.conv_bias = self.conv_bias.detach().cuda()        
            #print 'self size:',self.s  
            #if bias is None: # copy bias from current convolution.            
            self.bias.data.copy_(conv.bias.data[:s_bias])
        else:
            self.bias = None
            #else:
            #    self.bias = bias
                
        for p in conv.parameters():
            p.requires_grad = False
                    
    def setConvLearnable(self,T):
        for p in self.conv.parameters():
            p.requires_grad = T
    
    def set_bn(self,bn):
        my_bn = nn.BatchNorm2d(bn.num_features,affine=bn.affine)
        bn.eval()
        my_bn.load_state_dict(bn.state_dict())
        my_bn.train()
        self.my_bn = my_bn
        self.old_bn = bn
            
    def forward(self,x, alpha = None):
        # Modify the weights
        #conv = self.conv
        s = self.s
        w = self.w 
        if alpha is not None:
            #print 'got alpha'
            alpha1 = alpha.expand_as(w)
            newWeights = alpha1 * self.L(w) + (1-alpha1) * w
            if hasBias:
                alpha2 = alpha.squeeze().expand_as(self.bias)            
                bias = alpha2 * self.bias + (1-alpha2) * self.conv_bias
        else:            
            #print 'no alpha'
            newWeights = self.L(w)
            bias = self.bias
        newWeights = newWeights.transpose(0,1).contiguous()
        newWeights = newWeights.view(s)
                
        #print newWeights.size()
        #print bias.size()
        
        x = F.conv2d(x,newWeights,bias,stride=self.stride,padding=self.padding,dilation=self.dilation)
        
        # apply the batch normalization...
        if self.my_bn is not None:
            x_bn = self.my_bn(x)
            if alpha is not None:
                alpha3 = alpha.expand_as(x)
                x = alpha3 * x_bn + (1-alpha3) * self.old_bn(x)
            else:
                x = x_bn
        return x

def checkApproximation(net1,net2):
    a_orig = list(net1.features.children())
    a_fine = list(net2.features.children())
    abs_errors = []

    bar = progressbar.ProgressBar(max_value=len(a_fine)-1)
    for i,(orig,fine) in bar(enumerate(izip(a_orig,a_fine))):    
        if type(orig) is nn.BatchNorm2d:
            # make sure the batch-norm layers are unchanged
            ss1 = orig.state_dict()
            ss2 = fine.state_dict()
            assert ((ss1['running_mean']-ss2['running_mean']).sum()==0 and \
                    (ss1['running_var']-ss2['running_var']).sum()==0), \
            'found mismatch between batch norm on layer {}'.format(i)

            continue

        if type(orig) is not nn.Conv2d:
            continue
        s1 = orig.weight.size()
        nOrigParams = np.prod(s1)
        nNewParams = s1[0]*(1+s1[0])
        w1 = orig.weight.view(s1[0],-1) # Old weights
        s2 = fine.weight.size() 
        w2 = fine.weight.view(s1[0],-1) # new weights
        A = ton(w1).T        
        #A = A-np.mean(A,1,keepdims=True)
        B = ton(w2).T
        #B = A-np.mean(B,1,keepdims=True)
        X,residuals,rank,s = lstsq(A,B) # Approximation.
        cur_mean_error = np.abs((A.dot(X)-B)).mean()
        abs_errors.append(cur_mean_error)
    return abs_errors

    s1 = orig.weight.size()
    nOrigParams = np.prod(s1)
    nNewParams = s1[0]*(1+s1[0])
    w1 = orig.weight.view(s1[0],-1) # Old weights
    s2 = fine.weight.size() 
    w2 = fine.weight.view(s1[0],-1) # new weights
    A = ton(w1).T
    B = ton(w2).T                
    X,residuals,rank,s = lstsq(A,B) # Approximation.
    m = controlledConv(orig,torch.Tensor(X.T),fine.bias)
    return m,A,B,X

initializationTypes = ['linear_approx','random','diagonal']
def makeControlledConv(orig,fine,initializationType='linear_approx'):
    assert initializationType in initializationTypes,'Unknown initialization type from controlledConv: {}'.format(initializationType)
    s1 = orig.weight.size()
    s2 = fine.weight.size()
    nOrigParams = np.prod(s1)
    nNewParams = s2[0]*(1+s1[0])
    
    print s1,s2
    
    w1 = orig.weight.view(s1[0],-1) # Old weights
    s2 = fine.weight.size() 
    w2 = fine.weight.view(s2[0],-1) # new weights
    A = ton(w1).T
    B = ton(w2).T
    if initializationType == 'linear_approx':
        X,residuals,rank,s = lstsq(A,B) # Approximation.        
        
    elif initializationType == 'random':
        X = torch.zeros(s1[0],s2[0])
        #print '!!!!!!',X.size()
        init.xavier_uniform(X)
        X = X.numpy()
    elif initializationType == 'diagonal':
        # assert that s1 is a multiple of s2
        assert s1[0] % s2[0] == 0
        
        X = [torch.eye(s2[0])]* (s1[0] / s2[0])
        X = torch.cat(X)
        X = X.numpy()
        
    else:
        raise Exception('This code should not be reached.')
        
    m = controlledConv(orig,torch.Tensor(X.T),fine.bias)    
    
    return m,A,B,X

def makeControllerNetwork(net_orig,net_fine, initializationType='linear_approx', verbose = True, trackValues = True):
    """ Given two sequential networks net_orig and net_fine with the same structure,
        reformulate B so that is is compactly represented by re-using the weights of A. 
        Params :
            net_orig - the original network
            net_fine - network to be approximateed
            initializationType ['linear_approx']
            
            verbose - whether to track and print the layer-wise error for some random input, stemming 
            from the linear approximations.            
    """
    a_fine = list(net_fine.features.children())
    for p in net_fine.parameters():
        p.requires_grad=False

    a_orig = list(net_orig.features.children())
        
    v = Variable(torch.randn(1,3,64,64))
    v = v.cpu()

    value_fine = v.cuda()
    value_new = v.cuda()
    
    s_fine_vs_new = []
    s_controlled_vs_fine = []
    errors = []
    newChildren = []
    oldChildren = []
    types  = []
    
    #U = list(a_fine)
    bar = progressbar.ProgressBar(max_value=len(a_fine))
    
    for i,(orig,fine) in bar(enumerate(izip(a_orig,a_fine))): 
        wasBN = False
        #print i,
        tt = type(fine)        
        tt_str = str(tt)         
        types.append(tt_str.split('.')[-1][:-2])        
        if tt is nn.Conv2d:
            #if verbose: print '(conv)'
            #if use_linear_approx:
            m,A,B,X = makeControlledConv(orig,fine,initializationType)
            #else:
            #    m = controlledConv(orig,None)                          
            m.cuda() 
        elif tt is nn.BatchNorm2d:
            
            wasBN = True
            m.set_bn(orig)
            m = orig            
            #m = deepcopy(orig)                        
            #continue
        else:
            m = fine
            #if tt is nn.MaxPool2d:
                #if verbose: print '(maxpool2)'                    
            #elif tt is nn.ReLU:
            #    if verbose: print '(relu)'
                    
        value_fine_before = value_fine
        value_new_before = value_new
                        
        oldChildren.append(fine)
        if not wasBN:
            newChildren.append(m)            
        if trackValues:
            value_fine = fine(value_fine)
            value_new = m(value_new)
            curdiff = (value_fine-value_new).data.abs().mean()
            if verbose:
                print 'diff:',curdiff
        
            s_fine_vs_new.append(curdiff)
        
    return newChildren,oldChildren,s_fine_vs_new,types

def scalarVar(s):
    return Variable(torch.ones(1).cuda() * s)

def extractFeats(model,loader):
    # Extract all top-layer features once.
    cats = []
    feats = []
    for i,(a,b) in enumerate(loader):
        print i,
        a = Variable(a.cuda())
        feats.append(ton(model(a)))
        cats.append(b.numpy())
    feats = np.vstack(feats)
    cats = list(itertools.chain.from_iterable(cats))
    return feats,cats
def makeFeatLoader(model,loader,batch_size):
    feats,cats = extractFeats(model,loader)
    return DataLoader(TensorDataset(torch.Tensor(feats),torch.Tensor(cats)),batch_size=batch_size,shuffle=True)

class shifterNet(nn.Module):
    def __init__(self, decider,shiftable):
            super(shifterNet, self).__init__()
            self.decider = decider
            self.shiftable = shiftable            
    def forward(self, x):                                
        my_alpha = F.softmax(self.decider(x))[:,1:]
        my_alpha[my_alpha < .5] = 0
        my_alpha[my_alpha >= .5] = 1
        return self.shiftable(x,my_alpha)
    
'''
class Scale(object): # This is a copy from the torchvision repository, it's just a version conflict
    
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)
'''
def getTrainableParams(model):
    if type(model) is list:
        return [p for p in model if p.requires_grad]
    else:
        return [p for p in model.parameters() if p.requires_grad]
def makeTrainable(model,toggle):
    for p in model.parameters():
        p.requires_grad = toggle
    if hasattr(model,'features'):
        for q in model.features:
            q.train()

from PIL import Image,ImageOps
import numbers

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0, fill = 0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=self.fill)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))

# Make a relatively lightweight model for the baselines
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# Various configuration parameters

baseDataDir = os.path.expanduser('~/data_transfer/')
modelsBaseDir = os.path.expanduser('~/models')
all_datasets = {}
all_datasets['caltech256'] = {'trainDir': 'Caltech256/train/',
                          'testDir': 'Caltech256/test/',
                             'nClasses':257}                          
all_datasets['omniglot'] = {'trainDir': 'omniglot/python/train/',
                          'testDir': 'omniglot/python/test/',
                           'nClasses':1623}
all_datasets['daimler'] = {'trainDir': 'daimler/all_train/',
                          'testDir': 'daimler/all_test',
                          'nClasses':2}
all_datasets['sketch'] = {'trainDir': 'sketch_train',
                          'testDir': 'sketch_test',
                         'nClasses':250}
all_datasets['GTSR'] = {'trainDir': 'GTSR/Final_Training/',
                          'testDir': 'GTSR/Final_Test/',
                       'nClasses':43}
all_datasets['CIFAR-10'] = {'trainDir': 'cifar-10/train/',
                          'testDir': 'cifar-10/test/',
                       'nClasses':10}
all_datasets['CIFAR-100'] = {'trainDir': 'cifar-100/train/',
                          'testDir': 'cifar-100/test/',
                       'nClasses':100}
                           
all_datasets['SVHN'] = {'trainDir': 'svhn/train/',
                          'testDir': 'svhn/test/',
                       'nClasses':10}
all_datasets['plankton'] = {'trainDir': 'plankton_train',
                          'testDir': 'plankton_test',
                       'nClasses':121}
all_datasets['CUB'] = {'trainDir': 'CUB/train',
                          'testDir': 'CUB/test',
                       'nClasses':200}
all_datasets['mnist'] = {'trainDir': 'mnist/train',
                          'testDir': 'mnist/test',
                       'nClasses':10}

all_datasets_extra = {}
for k in all_datasets.keys():
    all_datasets_extra[k] = {}
#all_datasets_extra['sketch'] = {'crop_fill':1}
all_datasets_extra['SVHN'] = {'augment_flip':False}
all_datasets_extra['omniglot'] = {'augment_flip':False}

#dataset_stats = pickle.load(os.path.join(baseDataDir,'database_stats'))

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] # B
      
big_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] # D 

cuda=True
lr_drop_freq = 10
base_lr = 1e-3
adjust_learning_rate = None

import random
def makeNet(name,bigNet=False,fullyconv=False,batch_norm=True): 
    nClasses = all_datasets[name]['nClasses']
    my_cfg = cfg
    if bigNet:
        my_cfg = big_cfg
    model = VGG(make_layers(my_cfg,batch_norm=batch_norm,fullyconv=fullyconv),fc_size=2048, num_classes= nClasses,fullyconv=fullyconv)
    return model
'''
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image or np.ndarray with a probability of 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            if isinstance(img, np.ndarray):
                return np.fliplr(img)
            else:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    

'''
def makeLoaders2(name, stats = None):
    """
    quick and easy loaders,with default values
    """
    if stats is None:
        stats = dataset_stats[name]
    trainDir = os.path.join(baseDataDir,all_datasets[name]['trainDir'])
    testDir = os.path.join(baseDataDir,all_datasets[name]['testDir'])            
    augment_flip = all_datasets_extra[name].get("augment_flip", False)
    augment_crop = all_datasets_extra[name].get("augment_crop", False)
    crop_fill = all_datasets_extra[name].get("crop_fill", 
                                             tuple( (255*dataset_stats[name][0]).astype(np.uint8)))
    
    print augment_flip,augment_crop,crop_fill
        
    train_loader,test_loader = makeLoaders(trainDir,testDir,stats,augment_flip=augment_flip,
                                           augment_crop=augment_crop,crop_fill=crop_fill)
    
    return train_loader,test_loader

if False:
    def makeLoaders(train_dir,test_dir,stats,augment_flip=False, augment_crop = False,
                crop_fill = 0):
        # remove mean and divide by std (computed stats is variance, hence sqrt)
                    
        normalize = transforms.Normalize(np.asarray(stats[0]), np.asarray(stats[1])**.5)
        # random crop for jittering at train time.
        transform_list = [Scale((64,64))]
        if augment_crop:
            transform_list.append(RandomCrop(64,8,fill = crop_fill))
        if augment_flip:
            transform_list.append(RandomHorizontalFlip())
        
        transform_list.extend([transforms.ToTensor(), normalize])            
        transform_train=transforms.Compose(transform_list)

        db_train = DataLoader(dataset=datasets.ImageFolder(root = train_dir, transform=transform_train),
                    batch_size=128, shuffle=True,**kwargs)
        transform_test=transforms.Compose([Scale((64,64)), transforms.ToTensor(), normalize])
        db_test = DataLoader(dataset=datasets.ImageFolder(root = test_dir, transform=transform_test),
                    batch_size=128, shuffle=True)
        return db_train,db_test


def freezeAllButLastLayer(model):
    for p in model.features.parameters():
        p.requires_grad = False
    children = list(model.classifier.children())
    for p in children[:-1]:
        for q in p.parameters():
            q.requires_grad = False
    




def make_layers(cfg_1, batch_norm=False,instance_norm = False, affine=False,fullyconv=False):
    #print 'fully conv:',fullyconv
    cfg = list(cfg_1) # copy it to make sure it's not modified
    if batch_norm and instance_norm:
        raise Exception('cannot use both batch and instance normalization')
    layers = []
    in_channels = 3
    if fullyconv:
        cfg.append(512)
        #print cfg
    for i,v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            
            my_kernel_size = 3 # hacky!
            my_padding = 1
            if fullyconv and i == len(cfg)-1:
                #print '!'
                my_kernel_size = 2
                my_padding = 0
                            
            conv2d = nn.Conv2d(in_channels, v, kernel_size=my_kernel_size, padding=my_padding)            
            #init.kaiming_normal(conv2d.weight,mode='fan_out')
            init.kaiming_uniform(conv2d.weight)
            if batch_norm:
                layers +=[ conv2d, nn.BatchNorm2d(v, affine=affine), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v        
    return nn.Sequential(*layers)


def makeModelDirName(name,modelsBaseDir = modelsBaseDir, sfx='',baseNetwork=None):
    modelDir =  os.path.join(modelsBaseDir,'baseline_'+name+sfx)
    if baseNetwork is not None:
        modelDir += '_from_'+baseNetwork['name']
        if baseNetwork['onlyLastLayer']:
            modelDir+='_last'
    return modelDir
     
def doTrainingStuff(name,modelsBaseDir = modelsBaseDir, maxIters=np.inf, override=False,differentStats = None,
                   augment_flip=True,augment_crop = False, bigNet=False,
                   base_lr = 1e-3, baseNetwork=None,epochs=50,sfx='',batch_norm=True,
                   instance_norm=False, affine=False,cuda=True,lr_drop_freq=lr_drop_freq,optimizer=None,
                   adjust_learning_rate=adjust_learning_rate,disableBatchNorm=False,fullyconv=False):
        
    
    trainDir = os.path.join(baseDataDir,all_datasets[name]['trainDir'])
    testDir = os.path.join(baseDataDir,all_datasets[name]['testDir'])
        
    modelDir = makeModelDirName(name,modelsBaseDir = modelsBaseDir,sfx=sfx,baseNetwork=baseNetwork)
        
    if override:
        if not os.path.isdir(modelDir):
            print '{} : nothing to override - creating anew'.format(name)
        else:
            print 'warning - moving model for {} to backup at {}_BAK',name,modelDir
            shutil.move(modelDir,modelDir+'_BAK')
            
    if differentStats is not None:
        print 'using different stats...'
        stats = differentStats
    else:        
        print 
        stats = dataset_stats[name]
    
        
    augment_flip = all_datasets_extra[name].get("augment_flip", augment_flip)
    augment_crop = all_datasets_extra[name].get("augment_crop", augment_crop)
    crop_fill = all_datasets_extra[name].get("crop_fill", 
                                             tuple( (255*dataset_stats[name][0]).astype(np.uint8)))
        
    train_loader,test_loader = makeLoaders(trainDir,testDir,stats,augment_flip=augment_flip,
                                           augment_crop=augment_crop,crop_fill=crop_fill)
    #disableBatchNorm = False
    
    if baseNetwork is None:
        print 'training network from scratch...'
        #model = makeNet(name,bigNet)        
        
        nClasses = all_datasets[name]['nClasses']
        my_cfg = cfg
        if bigNet:
            my_cfg = big_cfg
        model = VGG(make_layers(my_cfg, batch_norm=batch_norm, instance_norm=instance_norm, affine=affine,fullyconv=fullyconv),
                    fc_size=2048, num_classes = nClasses,fullyconv=fullyconv)
        
    else: # fine tune from an existing network. 
        
        print 'fine tuning network from',baseNetwork['name']
        epochs = baseNetwork.get('max_ft_epochs',epochs)
        toContinue = baseNetwork.get('toContinue',False)
        model = baseNetwork['net']
        if not toContinue:
            mod = list(model.classifier.children())
            mod.pop()
            nClasses = all_datasets[name]['nClasses']
            mod.append(torch.nn.Linear(512, nClasses))
            model.classifier = nn.Sequential(*mod)
        control = baseNetwork.get('control',False)
        if control:
            raise NotImplementedError('Still need to link this to the controlling module')
        if baseNetwork['onlyLastLayer']:
            freezeAllButLastLayer(model)
            
        #disableBatchNorm = baseNetwork.get('disableBatchNorm',True)
        
                       
        # TODO - should we disallow the batch-norm layers to change from now?        
        #'onlyLastLayer':True,'max_ft_epochs':max_ft_epochs})
    params = [p for p in model.parameters() if p.requires_grad] 
    if cuda:
        model.cuda();
       
    if optimizer is None:
        optimizer = optim.Adam(params = params)
    elif type(optimizer) is str:
        if optimizer=='sgd':
            optimizer = optim.SGD(lr=base_lr,momentum=.9,weight_decay=0.0001,params = params)
        elif optimizer=='rmsprop':
            optimizer = optim.RMSprop(lr=base_lr,momentum=.9,weight_decay=0.0001,params = params)
        else:
            raise Exception('unexpected optimizer')
    
        
    trainAndTest(model= model, modelDir = modelDir, epochs=epochs, targetTranslator=None, model_save_freq=5,
                train_loader=train_loader, test_loader=test_loader, stopIfPerfect=True, optimizer=optimizer,
                criterion =nn.CrossEntropyLoss(), adjust_learning_rate=adjust_learning_rate,
                maxIters=maxIters,base_lr=base_lr, lr_drop_freq=lr_drop_freq, disableBatchNorm=disableBatchNorm,
                cuda=cuda)
    return model
def loadLastCheckpoint(model,modelDir,removeBest=False,verbose=False, onlyPerf = False):
    g = list(sorted(glob.glob(os.path.join(modelDir,'*'))))
    
    
    if verbose:
        print 'number of saved checkpoints:',len(g)
    
    if removeBest:
        g = [a for a in g if 'best' not in a]
    
    if len(g) > 0:
        lastCheckpoint = g[-1]
        if verbose:
            print 'last checkpoint:',lastCheckpoint
        
        #cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        #model = VGG(make_layers(cfg,True),fc_size=2048, num_classes= 1624)
        T = torch.load(lastCheckpoint)
        if not onlyPerf:
            if type(T) is dict:
                model.load_state_dict(T['state_dict'])
            else:
                model.load_state_dict(T)
            model.cuda();
        
        if verbose:            
            print 'loaded with accuracy of', T['best_acc']
        
        return T,model
    else:
        raise Exception('No checkpoint found for {}'.format(modelDir))    
        
def testNet(name,maxSamples=500,modelDir=None):
    trainDir = os.path.join(baseDataDir,all_datasets[name]['trainDir'])
    testDir = os.path.join(baseDataDir,all_datasets[name]['testDir'])
    
    if modelDir is None: 
        print 'reverting to default model dir.'
        modelDir =  os.path.join(modelsBaseDir,'baseline_'+name)
    train_loader,test_loader = makeLoaders(trainDir,testDir,dataset_stats[name])
    
    
    print 'testing dataset:',name
    
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    model = VGG(make_layers(cfg,True),fc_size=2048, num_classes= all_datasets[name]['nClasses'])            
    checkpoint = loadLastCheckpoint(model,modelDir)     
    return quickTest(model,test_loader,maxSamples=maxSamples)

def loadNet(name,verbose=False,isalpha=False):
    """
    loads the default network for this name
    """
    if verbose:
        print 'loading',name
    model = makeNet(name)
    modelDir =  makeModelDirName(name)
    checkpoint = loadLastCheckpoint(model,modelDir,verbose=verbose)    
    return model,checkpoint

# functions to "concatenate" the features parts neural networks.

# concatenate convolutions

def concatConv(c1,c2):
    newWeights = torch.cat([c1.weight,c2.weight])
    newBias = torch.cat([c1.bias,c2.bias])
    s = c1.weight.size()
    c3 = nn.Conv2d(s[1],2*s[0],s[2],stride = c1.stride, padding=c1.padding)
    c3.weight.data = newWeights.data
    c3.bias.data = newBias.data
    return c3


def concatBN(bn1,bn2):
    s = bn1.num_features
    bn3 = nn.BatchNorm2d(s*2)
    for k in ['running_mean','running_var']:
        bn3.state_dict()[k] = torch.cat([bn1.state_dict()[k],bn1.state_dict()[k]])
    return bn3


# good!
def concatNets(net1,net2): # concatenet :-)
    newFeatures = []
    for f1,f2 in zip(net1.features,net2.features):
        tt1,tt2 = type(f1),type(f2)        
        assert tt1==tt2,'cannot concatenate networks with different structures'        
        if tt1 is nn.Conv2d:
            newFeatures.append(concatConv(f1,f2))
        elif tt1 is nn.BatchNorm2d:
            newFeatures.append(concatBN(f1,f2))
        elif tt1 in [nn.MaxPool2d, nn.ReLU]:
            newFeatures.append(f1)
        else:
            Exception('Don''t know how to "concatenate" modules of types {},{}'.format(tt1,tt2))
    return newFeatures

'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append( (impath, int(imlabel)) )

    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
            flist_reader=default_flist_reader, loader=default_loader):
            self.root   = root
            self.imlist = flist_reader(flist)		
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root,impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)
