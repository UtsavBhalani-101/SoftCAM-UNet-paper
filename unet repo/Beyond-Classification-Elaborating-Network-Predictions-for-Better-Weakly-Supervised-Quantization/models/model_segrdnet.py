import torch
import torch.nn as nn
from functools import partial
from typing import List
from torchvision import models
from models.rdnet import *
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.squeeze_excite import EffectiveSEModule
from timm.models import create_model, register_model, build_model_with_cfg, named_apply, generate_default_cfgs
from timm.models.layers import DropPath
from timm.models.layers import LayerNorm2d
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


class AVG_POOL(nn.Module):
    """Average pooling replacement for LOG_SUM_EXP"""
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.check = nn.Identity()
    def forward(self, x):
        # x: (B, C, H, W) -> (B, C)
        x = self.check(x)
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)


class SoftCAMHead(nn.Module):
    """
    SoftCAM-style evidence head for a decoder stage.
    Produces per-class evidence maps and global predictions.
    Includes L1/L2 regularization on evidence maps (SoftCAM training objective).
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # SoftCAM-style classifier: 1x1 conv producing per-class evidence maps
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Global average pooling for classification logits
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Store evidence maps for regularization
        self.evidence_maps = None
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature maps from decoder
        Returns:
            logits: (B, num_classes) - global predictions
            evidence_maps: (B, num_classes, H, W) - per-class evidence maps
        """
        # Produce per-class evidence maps
        evidence_maps = self.classifier(x)  # (B, num_classes, H, W)
        
        # Store for regularization loss
        self.evidence_maps = evidence_maps
        
        # Global average pooling for classification
        logits = self.avgpool(evidence_maps).flatten(1)  # (B, num_classes)
        
        return logits, evidence_maps
    
    def get_regularization_loss(self, lambda_l1=0.0, lambda_l2=0.0):
        """
        Compute L1/L2 regularization on evidence maps (SoftCAM training objective).
        """
        if self.evidence_maps is None:
            return 0.0
        
        l1_loss = torch.norm(self.evidence_maps, p=1) if lambda_l1 > 0 else 0.0
        l2_loss = torch.norm(self.evidence_maps, p=2) if lambda_l2 > 0 else 0.0
        
        return lambda_l1 * l1_loss + lambda_l2 * l2_loss


class SoftCAMFeatureRefiner(nn.Module):
    """Compact feature-refinement branch used by Decoder 0 Option B."""
    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SoftCAMDecoderStage(nn.Module):
    """Full Decoder 0 replacement with evidence and next-stage feature outputs."""
    def __init__(self, in_channels: int, num_classes: int, feature_channels: int = 256):
        super().__init__()
        self.refiner = SoftCAMFeatureRefiner(in_channels, feature_channels)
        self.softcam_head = SoftCAMHead(feature_channels, num_classes)
        self.to_next_stage = nn.Sequential(
            nn.Conv2d(feature_channels, in_channels, kernel_size=1, bias=False),
            LayerNormChan(in_channels, data_format="channels_first"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        refined_features = self.refiner(x)
        logits, evidence_maps = self.softcam_head(refined_features)
        features_for_next_stage = self.to_next_stage(refined_features)
        return logits, evidence_maps, features_for_next_stage

    def get_regularization_loss(self, lambda_l1=0.0, lambda_l2=0.0):
        return self.softcam_head.get_regularization_loss(lambda_l1, lambda_l2)


# Original LOG_SUM_EXP and AVG_POOL classes preserved for other decoders





class RDNetClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        drop_rate: float = 0.,
        akorn_on: bool = False,
        pooling: str = 'avg',  # 'lse' or 'avg'
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
        
        # Pooling selection
        if pooling == 'lse':
            self.wt_exp = LOG_SUM_EXP(r=self.r)
        else:  # 'avg'
            self.wt_exp = AVG_POOL(num_classes)

        print(f'Pooling: {pooling}')
    def reset(self, num_classes):
        self.fc = nn.Conv2d(self.num_features, num_classes, 1, padding = 0) if num_classes > 0 else nn.Identity()   
        if hasattr(self.wt_exp, 'num_classes'):
            self.wt_exp.num_classes = num_classes

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
        init_checkpoint = False,
        pooling: str = 'avg',  # 'lse' or 'avg'
        use_softcam_decoder0: bool = False,  # Option A: SoftCAM evidence head for Decoder 0
        softcam_decoder0_mode: str = 'a',  # 'a': head, 'b': side-branch refiner, 'c': replace stage
        softcam_lambda_l1: float = 0.0,
        softcam_lambda_l2: float = 0.0,
    ):
        super().__init__()

        self.model = create_model( 'rdnet_tiny', pretrained= False, in_chans=3, num_classes=num_classes, drop_rate=0.0, drop_path_rate= 0, drop_block_rate=0, global_pool=None,)
        self.activation = {}
        self.name_dict = {'dense_stages.0':3, 'dense_stages.1':2, 'dense_stages.5':1, 'dense_stages.6':0}
        self.init_forward_hook()
        if init_checkpoint:
            self.init_checkpoint()
        ch_list = [1040,744,440,256]
        self.head = RDNetClassifierHead(embedding_dim, num_classes, pooling=pooling)
        self.fc0 = nn.Conv2d(ch_list[0], num_classes,1,padding=0) 
        self.fc1 = nn.Conv2d(ch_list[1], num_classes,1,padding=0) 
        self.fc2 = nn.Conv2d(ch_list[2], num_classes,1,padding=0) 
        self.fc3 = nn.Conv2d(ch_list[3], num_classes,1,padding=0) 
        
        # Pooling selection for multi-scale outputs
        if pooling == 'lse':
            self.lse0 = LOG_SUM_EXP(num_classes)
            self.lse1 = LOG_SUM_EXP(num_classes)
            self.lse2 = LOG_SUM_EXP(num_classes)
            self.lse3 = LOG_SUM_EXP(num_classes)
        else:  # 'avg'
            self.lse0 = AVG_POOL(num_classes)
            self.lse1 = AVG_POOL(num_classes)
            self.lse2 = AVG_POOL(num_classes)
            self.lse3 = AVG_POOL(num_classes)

        self.seq0 = None if (use_softcam_decoder0 and softcam_decoder0_mode == 'c') else nn.Sequential(EffectiveSEModule(ch_list[0]))
        self.up0 = nn.Sequential(nn.Conv2d(ch_list[0], ch_list[1],1,padding=0),DepthToSpace2DWithConv(ch_list[1], 2, 2),LayerNormChan(ch_list[1], data_format="channels_first"))
        self.seq1 = nn.Sequential(nn.Conv2d(2*ch_list[1], ch_list[1],1,padding=0), LayerNormChan(ch_list[1], data_format="channels_first") ,nn.ReLU(),EffectiveSEModule(ch_list[1]))
        self.up1 = nn.Sequential(nn.Conv2d(ch_list[1],ch_list[2],1,padding=0),DepthToSpace2DWithConv(ch_list[2], 2, 2),LayerNormChan(ch_list[2], data_format="channels_first"))               
        self.seq2 = nn.Sequential(nn.Conv2d(2*ch_list[2], ch_list[2],1,padding=0), LayerNormChan(ch_list[2], data_format="channels_first") ,nn.ReLU(),EffectiveSEModule(ch_list[2]))
        self.up2 = nn.Sequential(nn.Conv2d(ch_list[2], ch_list[3],1,padding=0),DepthToSpace2DWithConv(ch_list[3], 2, 2),LayerNormChan(ch_list[3], data_format="channels_first"))               
        self.seq3 = nn.Sequential(nn.Conv2d(2*ch_list[3], ch_list[3],1,padding=0),LayerNormChan(ch_list[3], data_format="channels_first") ,nn.ReLU(),EffectiveSEModule(ch_list[3]))
        self.check = nn.Identity()

        # Option A: SoftCAM evidence head for Decoder 0
        self.use_softcam_decoder0 = use_softcam_decoder0
        if softcam_decoder0_mode not in {'a', 'b', 'c'}:
            raise ValueError("softcam_decoder0_mode must be 'a', 'b', or 'c'")
        self.softcam_decoder0_mode = softcam_decoder0_mode
        self.softcam_lambda_l1 = softcam_lambda_l1
        self.softcam_lambda_l2 = softcam_lambda_l2
        
        if use_softcam_decoder0:
            if softcam_decoder0_mode == 'c':
                # Option C replaces seq0 and emits a new compatible feature
                # tensor for Decoder 1, alongside the class evidence map.
                self.softcam_decoder0 = SoftCAMDecoderStage(ch_list[0], num_classes)
            elif softcam_decoder0_mode == 'b':
                # Option B retains the original feature path and refines only
                # the separate evidence branch.
                self.softcam_refiner0 = SoftCAMFeatureRefiner(ch_list[0])
                softcam_channels = 256
                self.softcam_decoder0 = SoftCAMHead(softcam_channels, num_classes)
            else:  # Option A
                softcam_channels = ch_list[0]
                self.softcam_decoder0 = SoftCAMHead(softcam_channels, num_classes)

    def get_softcam_regularization(self):
        """Get SoftCAM regularization loss from Decoder 0 evidence maps."""
        if self.use_softcam_decoder0:
            if self.softcam_decoder0_mode == 'c':
                return self.softcam_decoder0.get_regularization_loss(
                    self.softcam_lambda_l1, self.softcam_lambda_l2
                )
            return self.softcam_decoder0.get_regularization_loss(self.softcam_lambda_l1, self.softcam_lambda_l2)
        return 0.0


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
        
        if self.use_softcam_decoder0 and self.softcam_decoder0_mode == 'c':
            # Option C replaces the original seq0 stage. It returns a new
            # 1,040-channel feature map for up0/Decoder 1 and a class map.
            l_out_0, f_out_0, middle_0 = self.softcam_decoder0(outputs[0])
        else:
            middle_0 = self.seq0(outputs[0])
            if self.use_softcam_decoder0:
                # Options A/B preserve the original Decoder 0 feature path.
                softcam_features_0 = middle_0
                if self.softcam_decoder0_mode == 'b':
                    softcam_features_0 = self.softcam_refiner0(middle_0)
                l_out_0, f_out_0 = self.softcam_decoder0(softcam_features_0)
            else:
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
        
        # For Option A, f_out_0 is the SoftCAM evidence map; otherwise it is
        # the original fc0 map. Both remain class-channel tensors for MSE.
        mse_loss = [ 
            F.mse_loss(F.adaptive_avg_pool2d(f_out_1, (16,16)), f_out_0),
            F.mse_loss(F.adaptive_avg_pool2d(f_out_2, (32,32)), f_out_1),
            F.mse_loss(F.adaptive_avg_pool2d(f_out_3, (64,64)), f_out_2)
        ][::-1]
        
        if not eval_mode:
            return l_out_3  , mse_loss , (l_out_3, l_out_2, l_out_1, l_out_0)
        else:
            return l_out_3
