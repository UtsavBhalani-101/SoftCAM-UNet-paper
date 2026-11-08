import torch
import torch.nn as nn
from collections import OrderedDict

## Classification head based on convolution as it is used in the revised bagnet architecture
class ConvClassifier(nn.Module):
    def __init__(self, num_ftrs, n_classes):
        super(ConvClassifier, self).__init__()
        self.num_ftrs = num_ftrs
        self.n_classes = n_classes
        self.sparse = nn.Conv2d(
            self.num_ftrs, 
            self.n_classes, 
            kernel_size=(1, 1), 
            stride=1
        )
    
    def forward(self, x):
        return self.sparse(x)


## From blackbox to inherently explainable model
class FullyConvModel(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super(FullyConvModel, self).__init__()

        network = cfg.train.network
       
        if 'resnet' in network or 'resnext' in network or 'bagnet' in network:
            backbone = list(model.children())[:-2]
            classifier = nn.Conv2d(model.fc.in_features, num_classes, kernel_size=(1,1), stride=1)
        elif 'vgg' in network:  
            backbone = list(model.children())[:-2]
            before_cls = nn.Sequential(
                nn.Conv2d(512, 4096, kernel_size=(1,1), stride=1),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Conv2d(4096, 4096, kernel_size=(1,1), stride=1),
                nn.ReLU(True),
                nn.Dropout()
            )
            backbone.append(before_cls)
            classifier = nn.Conv2d(4096, num_classes, kernel_size=(1,1), stride=1)
        elif 'densenet' in network:
            backbone = list(model.children())[:-1]
            classifier = nn.Conv2d(model.classifier.in_features, num_classes, kernel_size=(1,1), stride=1)
        elif 'efficientnet' in network:
            backbone = list(model.children())[:-2]
            backbone.append(nn.Dropout(0.3)),
            classifier = nn.Conv2d(model.classifier[-1].in_features, num_classes, kernel_size=(1,1), stride=1)
        elif 'inception' in network:
            backbone = list(model.children())[:-3]
            classifier = nn.Conv2d(model.last_linear.in_features, num_classes, kernel_size=(1,1), stride=1)
        else:
            raise NotImplementedError('Not implemented network.')
            
        self.backbone = nn.Sequential(*backbone)
        self.classifier = classifier

    def forward(self, x):        
        x = self.backbone(x)                # (bs, c, h, w) 
        activation = self.classifier(x)     # (bs, n_class, h, w) 
        bs, c, h, w = x.shape               # (bs, n_class, h, w) 
          
        avgpool = nn.AvgPool2d(kernel_size=(h, w), stride=(1,1), padding=0)             
        out = avgpool(activation)           # (bs, n_class, 1, 1) 
        out = out.view(out.shape[0], -1)    # (bs, n_class)
        
        return out, activation


## Sparse BagNet
class SparseBagnet(nn.Module):
    def __init__(self, model, num_classes):
        super(SparseBagnet, self).__init__()

        last_block = list(model.layer4.children())[-1] 
        num_channels = last_block.conv3.out_channels

        backbone = list(model.children())[:-2]
        self.backbone = nn.Sequential(*backbone)
        
        # classification layer: conv2d instead of FCL
        self.classifier = nn.Conv2d(num_channels, num_classes, kernel_size=(1,1), stride=1)
        # 24 ~ 224, 60~ 512
        self.clf_avgpool = nn.AvgPool2d(kernel_size=(1,1), stride=(1,1), padding=0)

    def forward(self, x):        
        x = self.backbone(x)                # (bs, c, h, w) 
        activation = self.classifier(x)     # (bs, n_class, h, w) 
        bs, c, h, w = x.shape               # (bs, n_class, h, w) 
          
        avgpool = nn.AvgPool2d(kernel_size=(h, w), stride=(1,1), padding=0)             
        out = avgpool(activation)           # (bs, n_class, 1, 1) 
        out = out.view(out.shape[0], -1)    # (bs, n_class)

        att_weight = torch.zeros((bs, c, h, w), device='cuda')
        
        return out, activation, att_weight