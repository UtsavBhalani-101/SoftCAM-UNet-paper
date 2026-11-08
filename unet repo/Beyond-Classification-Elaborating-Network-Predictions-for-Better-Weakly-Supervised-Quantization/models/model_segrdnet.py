import torch
import torch.nn as nn 
from functools import partial
from typing import List
from torchvision import models
from models.rdnet import *
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.squeeze_excite import EffectiveSEModule
from timm.models import create_model,register_model, build_model_with_cfg, named_apply, generate_default_cfgs
from timm.models.layers import DropPath
from timm.models.layers import LayerNorm2d

from timm.layers.squeeze_excite import EffectiveSEModule
import torch.nn.functional as F


# follow https://github.com/HelmholtzAI-FZJ/flex_gen/blob/main/flex_gen/autoencoders/flexTokenizer/layers.py
# from open-magvit
def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Depth-to-Space DCR mode (depth-column-row) core implementation.

    Args:
        x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
        block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)

    return x


class DepthToSpace2DWithConv(nn.Module):
    def __init__(
        self,
        channels_in,
        block_size_h,
        block_size_w,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels_in,
            block_size_h * block_size_w * channels_in,
            kernel_size=3,
            padding=1,
        )
        # nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        self.block_size_h = block_size_h
        self.block_size_w = block_size_w

    def forward(self, x):
        x = self.conv(x)
        # x = depth_to_space_2d(x, self.block_size_h, self.block_size_w)
        x = depth_to_space(x, self.block_size_h)
        return x



## Taken from ConvNeXt: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class LayerNormChan(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



def log_sum_exp(out, r):
    if len(out.size()) > 3:
        out = out.flatten(start_dim=2)
    out = torch.permute(out,(0,2,1))
    b, c, w = out.size()
    multi = r* torch.ones_like(out)
    out_max = torch.max(out, dim=1)[0].unsqueeze(1)
    out =     out_max.squeeze(1) + torch.div( torch.log(torch.mean(torch.exp(multi*(out-out_max)),dim=1)), torch.mean(1e-8+ multi,dim=1))
            
    return out


class LOG_SUM_EXP(nn.Module):
    def __init__(
        self,            
        r: float = 3.,
        num_classes: int = 6,
    ):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros((num_classes)), requires_grad=True)
        self.r = r
        self.check = nn.Identity()
    def forward(self, x):
        _, c , _ ,_ = x.size()
        x = x - self.bias[None,:, None,None]
        x = self.check(x)
        final = log_sum_exp(x, self.r)

        return final





class RDNetClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        drop_rate: float = 0.,
        akorn_on: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_features = in_features

        #self.norm = LayerNormChan(in_features, data_format="channels_first")  #nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Conv2d(self.num_features, num_classes, 1, padding = 0) if num_classes > 0 else nn.Identity() 
        self.prenorm = LayerNormChan(in_features, data_format="channels_first")

        self.se = EffectiveSEModule(in_features)
        self.r = 3
        self.wt_exp = LOG_SUM_EXP(r= self.r)

        print('r value:', self.r, 'use avg')
    def reset(self, num_classes):
        self.fc = nn.Conv2d(self.num_features, num_classes, 1, padding = 0) if num_classes > 0 else nn.Identity()   

    def feature(self, x):
        x = self.prenorm(x)
        return x
    
    def forward(self, x, pre_logits: bool = False, r=3):
        x = self.feature(x)

        x = self.drop(x)
        if pre_logits:
            return x
        x = self.se(x)  
        x = self.fc(x)
        x = self.wt_exp(x)
        return x



class SegRDNet(nn.Module):
    def __init__(
        self,
        embedding_dim = 1024,
        num_classes = 6,
        init_checkpoint = False
    ):
        super().__init__()

        self.model = create_model( 'rdnet_tiny', pretrained= False, in_chans=3, num_classes=num_classes, drop_rate=0.0, drop_path_rate= 0, drop_block_rate=0, global_pool=None,)
        self.activation = {}
        self.name_dict = {'dense_stages.0':3, 'dense_stages.1':2, 'dense_stages.5':1, 'dense_stages.6':0}
        self.init_forward_hook()
        if init_checkpoint:
            self.init_checkpoint()
        ch_list = [1040,744,440,256]
        self.head = RDNetClassifierHead(embedding_dim, num_classes)
        self.fc0 = nn.Conv2d(ch_list[0], num_classes,1,padding=0) 
        self.fc1 = nn.Conv2d(ch_list[1], num_classes,1,padding=0) 
        self.fc2 = nn.Conv2d(ch_list[2], num_classes,1,padding=0) 
        self.fc3 = nn.Conv2d(ch_list[3], num_classes,1,padding=0) 
        
        self.lse0 = LOG_SUM_EXP(num_classes)
        self.lse1 = LOG_SUM_EXP(num_classes)
        self.lse2 = LOG_SUM_EXP(num_classes)
        self.lse3 = LOG_SUM_EXP(num_classes)

        self.seq0 = nn.Sequential(EffectiveSEModule(ch_list[0])) 
        self.up0 = nn.Sequential(nn.Conv2d(ch_list[0], ch_list[1],1,padding=0),DepthToSpace2DWithConv(ch_list[1], 2, 2),LayerNormChan(ch_list[1], data_format="channels_first"))
        self.seq1 = nn.Sequential(nn.Conv2d(2*ch_list[1], ch_list[1],1,padding=0), LayerNormChan(ch_list[1], data_format="channels_first") ,nn.ReLU(),EffectiveSEModule(ch_list[1]))
        self.up1 = nn.Sequential(nn.Conv2d(ch_list[1],ch_list[2],1,padding=0),DepthToSpace2DWithConv(ch_list[2], 2, 2),LayerNormChan(ch_list[2], data_format="channels_first"))               
        self.seq2 = nn.Sequential(nn.Conv2d(2*ch_list[2], ch_list[2],1,padding=0), LayerNormChan(ch_list[2], data_format="channels_first") ,nn.ReLU(),EffectiveSEModule(ch_list[2]))
        self.up2 = nn.Sequential(nn.Conv2d(ch_list[2], ch_list[3],1,padding=0),DepthToSpace2DWithConv(ch_list[3], 2, 2),LayerNormChan(ch_list[3], data_format="channels_first"))               
        self.seq3 = nn.Sequential(nn.Conv2d(2*ch_list[3], ch_list[3],1,padding=0),LayerNormChan(ch_list[3], data_format="channels_first") ,nn.ReLU(),EffectiveSEModule(ch_list[3]))
        self.check = nn.Identity()


    def init_forward_hook(self):
        for (name, module) in self.model.named_modules():
            if name in  ['dense_stages.6', 'dense_stages.5', 'dense_stages.1', 'dense_stages.0']:
                module.register_forward_hook(self.get_activation(name))

    def init_checkpoint(self):
        checkpoint = torch.load('./output/train/20250906-095556-rdnet_tiny-512/checkpoint-2.pth.tar') 
        self.model.load_state_dict(checkpoint['state_dict'])


    def get_activation(self,name):
        def hook(model, input_, output):
            self.activation[self.name_dict[name]] = output
        return hook

    def forward(self, x, eval_mode= True):
        x = self.model(x)

        outputs = [self.activation[i] for i in range(4)]
        middle_0 = self.seq0(outputs[0])
        f_out_0 = self.fc0(middle_0)
        f_out_0 = self.check(f_out_0)
        l_out_0 = self.lse0(f_out_0)
        middle_1 = self.seq1(torch.cat((outputs[1], self.up0(middle_0)),dim =1 ))
        f_out_1 = self.fc1(middle_1)
        l_out_1 = self.lse1(f_out_1)
        middle_2 = self.seq2(torch.cat((outputs[2], self.up1(middle_1)),dim =1 ))
        f_out_2 = self.fc2(middle_2)
        l_out_2 = self.lse2(f_out_2)
        middle_3 = self.seq3(torch.cat((outputs[3], self.up2(middle_2)),dim =1 ))
        f_out_3 = self.fc3(middle_3)
        l_out_3 = self.lse3(f_out_3)
        mse_loss = [ F.mse_loss(F.adaptive_avg_pool2d(f_out_1, (16,16)) , f_out_0 ), F.mse_loss(F.adaptive_avg_pool2d(f_out_2, (32,32)) , f_out_1 ), F.mse_loss(F.adaptive_avg_pool2d(f_out_3, (64,64)) , f_out_2 )][::-1]
        if not eval_mode:
            return l_out_3  , mse_loss , (l_out_3, l_out_2, l_out_1, l_out_0)
        else:
            return l_out_3
