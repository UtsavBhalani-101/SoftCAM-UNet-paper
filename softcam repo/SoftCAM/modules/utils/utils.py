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

## MHSA
class MHSA(nn.Module):
    def __init__(self, in_dim, heads=8):
        super(MHSA, self).__init__()
        self.heads = heads
        self.scale = (in_dim // heads) ** -0.5
        self.norm1 = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.to_out = nn.Linear(in_dim, in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(0.1)  # Optional dropout

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [b, c, h*w] => [b, h*w, c]
        residual = x
        x = self.norm1(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1) # chunk create tuple with Q, K V
        q, k, v = map(lambda t: t.reshape(b, h*w, self.heads, -1).transpose(1, 2), qkv)
        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, h*w, -1)
        out = self.to_out(out)
        out = self.dropout(out)  # Optional dropout
        out = out + residual  # First residual connection
        residual = out
        out = self.norm2(out)
        out = out + residual  # Second residual connection
        return out.transpose(1, 2).reshape(b, c, h, w)

# FCT ~ Samuel
class MHSAConv(nn.Module):
    def __init__(self, in_dim, heads=8):
        super(MHSAConv, self).__init__()
        self.heads = heads
        self.scale = (in_dim // heads) ** -0.5

        # Replace nn.LayerNorm with nn.GroupNorm as it is better than LayerNorm when it comes to spatial data
        self.norm1 = nn.GroupNorm(1, in_dim)
        self.norm2 = nn.GroupNorm(1, in_dim)

        # Replace nn.Linear with nn.Conv2d
        self.to_qkv = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1, bias=True)
        self.to_out = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.dropout = nn.Dropout(0.1)  # Optional dropout

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm1(x)

        # Compute qkv
        qkv = self.to_qkv(x)  # [b, c*3, h, w]
        q, k, v = qkv.chunk(3, dim=1)  # Each is [b, c, h, w]

        # Reshape q, k, v for attention computation
        head_dim = c // self.heads
        q = q.view(b, self.heads, head_dim, h * w).permute(0, 1, 3, 2)  # [b, heads, h*w, head_dim]
        k = k.view(b, self.heads, head_dim, h * w).permute(0, 1, 3, 2)
        v = v.view(b, self.heads, head_dim, h * w).permute(0, 1, 3, 2)

        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [b, heads, h*w, h*w]
        attn = dots.softmax(dim=-1)  # [b, heads, h*w, h*w]

        # Apply attention to v
        out = torch.matmul(attn, v)  # [b, heads, h*w, head_dim]

        # Reshape back to [b, c, h, w]
        out = out.permute(0, 1, 3, 2).contiguous()  # [b, heads, head_dim, h*w]
        out = out.view(b, c, h, w)

        # Output projection
        out = self.to_out(out)
        out = self.dropout(out)

        # Residual connection
        out = out + residual
        out = self.norm2(out)
        residual = out
        out = out + residual
        return out

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

## From blackbox to inherently explainable model
class FullyConvModel(nn.Module):
    def __init__(self, model, num_classes):
        super(FullyConvModel, self).__init__()

        last_block = list(model.layer4.children())[-1] 
        num_channels = last_block.conv3.out_channels

        backbone = list(model.children())[:-2]
        self.backbone = nn.Sequential(*backbone)
        
        # classification layer: conv2d instead of FCL
        self.classifier = nn.Conv2d(num_channels, num_classes, kernel_size=(1,1), stride=1)
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

## backbone for the mutitask class
class Backbone(nn.Module):
    def __init__(self, model):
        super(Backbone, self).__init__()
        self.num_ftrs = model.fc.in_features
        self.body = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.body(x)
    
## Conv classification layer
class ConvPenultimate(nn.Module):
    def __init__(self, num_ftrs, n_classes, cfg=None):
        super(ConvPenultimate, self).__init__()
        self.cfg = cfg
        self.classifier = nn.Conv2d(num_ftrs, n_classes, kernel_size=(1, 1), stride=1)
        
    def forward(self, x):
        heatmap = self.classifier(x)
        bs, c, h, w = x.shape
        self.avgpool = nn.AvgPool2d(kernel_size=(h, w), stride=(1, 1), padding=0)
        out = self.avgpool(heatmap)
        out = out.view(out.shape[0], -1)
        if self.cfg.train.train_with_att:
            att_weight = x
        else:
            att_weight = torch.zeros((bs, c, h, w), device='cuda')
        return out, heatmap, att_weight

## FCL classification layer
class FCLPenultimate(nn.Module):
    def __init__(self, num_ftrs, n_classes, cfg=None):
        super(FCLPenultimate, self).__init__()
        self.cfg = cfg
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_ftrs, n_classes)
        
    def forward(self, x):
        bs, c, h, w = x.shape
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  
        out = self.classifier(x)
        heatmap = torch.zeros((bs, 2, h, w), device='cuda')
        if self.cfg.train.train_with_att:
            att_weight = x
        else:
            att_weight = torch.zeros((bs, c, h, w), device='cuda')
        return out, heatmap, att_weight

## Multitask model 
class MultiTaskModel(nn.Module):
    def __init__(self, model, fcl_cls , n_classes, cfg):
        super(MultiTaskModel, self).__init__()
        self.body = Backbone(model)
        num_ftrs = self.body.num_ftrs
        #['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        if fcl_cls:
            self.pen_Atel = ConvPenultimate(num_ftrs, n_classes, cfg=cfg)
            self.pen_Card = ConvPenultimate(num_ftrs, n_classes, cfg=cfg)
            self.pen_Cons = ConvPenultimate(num_ftrs, n_classes, cfg=cfg)
            self.pen_Edem = ConvPenultimate(num_ftrs, n_classes, cfg=cfg)
            self.pen_Pleu = ConvPenultimate(num_ftrs, n_classes, cfg=cfg)
        else:
            self.pen_Atel = FCLPenultimate(num_ftrs, n_classes, cfg=cfg)
            self.pen_Card = FCLPenultimate(num_ftrs, n_classes, cfg=cfg)
            self.pen_Cons = FCLPenultimate(num_ftrs, n_classes, cfg=cfg)
            self.pen_Edem = FCLPenultimate(num_ftrs, n_classes, cfg=cfg)
            self.pen_Pleu = FCLPenultimate(num_ftrs, n_classes, cfg=cfg)

    def forward(self, x):
        x = self.body(x)
        return self.pen_Atel(x), self.pen_Card(x), self.pen_Cons(x), self.pen_Edem(x), self.pen_Pleu(x)
