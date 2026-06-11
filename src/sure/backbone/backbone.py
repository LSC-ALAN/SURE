import torch.nn as nn
import torch
from src.sure.backbone.convnextv2.convnextv2 import convnextv2_nano
from einops import rearrange
from kornia.utils import create_meshgrid

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=bias,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNet18(nn.Module):
    """
    Fewer channels
    """

    def __init__(self, config=None):
        super().__init__()
        # Config
        block_dims = config["backbone"]["block_dims"]
        self.lin_4 = nn.Conv2d(64, 128, 1)
        self.lin_8 = nn.Conv2d(128, 256, 1)
        # Networks
        self.conv1 = nn.Conv2d(
            3, block_dims[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(block_dims[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            BasicBlock, block_dims[0], block_dims[0], stride=1
        )  # 1/2
        self.layer2 = self._make_layer(
            BasicBlock, block_dims[0], block_dims[1], stride=2
        )  # 1/4
        self.layer3 = self._make_layer(
            BasicBlock, block_dims[1], block_dims[2], stride=2
        )  # 1/8
        # self.layer4 = self._make_layer(
        #     BasicBlock, block_dims[2], block_dims[3], stride=2
        # )  # 1/16
        # self.layer5 = self._make_layer(
        #     BasicBlock, block_dims[3], block_dims[4], stride=2
        # )  # 1/32

        # For fine matching
        # self.fine_conv = nn.Sequential(
        #     self._make_layer(
        #         BasicBlock, block_dims[2], block_dims[2], stride=1),
        #     conv1x1(block_dims[2], block_dims[4]),
        #     nn.BatchNorm2d(block_dims[4]),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_dim, out_dim, stride=1):
        layer1 = block(in_dim, out_dim, stride=stride)
        layer2 = block(out_dim, out_dim, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, data):
        B, _, H, W = data['image0'].shape
        x = torch.cat([data['image0'], data['image1']], 0)
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        
        feat_8_0, feat_8_1 = self.lin_8(x3).split(B)
        feat_4_0, feat_4_1 = self.lin_4(x2).split(B)

        scale = 8
        h_8, w_8 = H // scale, W // scale
        device = data['image0'].device
        grid = [rearrange((create_meshgrid(h_8, w_8, False, device) * scale).squeeze(0),
                          'h w t->(h w) t')] * B  # kpt_xy
        grid_8 = torch.stack(grid, 0)

        data.update({
            'bs': B,
            'c': feat_8_0.shape[1],
            'h_8': h_8,
            'w_8': w_8,
            'hw_8': h_8 * w_8,
            'feat_8_0': feat_8_0,
            'feat_8_1': feat_8_1,
            'feat_4_0': feat_4_0,
            'feat_4_1': feat_4_1,
            'grid_8': grid_8,
        })
        # x0 = self.relu(self.bn1(self.conv1(x)))
        # x1 = self.layer1(x0)  # 1/2
        # x2 = self.layer2(x1)  # 1/4
        # x3 = self.layer3(x2)  # 1/8
        # # x4 = self.layer4(x3)  # 1/16
        # # x5 = self.layer5(x4)  # 1/32

        # # xf = self.fine_conv(x3)  # 1/8

        # return [x3, x4, x5, xf]




class CovNextV2_nano(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = convnextv2_nano()
        self.cnn.norm = None
        self.cnn.head = None
        self.cnn.downsample_layers[2] = None
        self.cnn.downsample_layers[3] = None
        self.cnn.stages[2] = None
        self.cnn.stages[3] = None

        self.lin_4 = nn.Conv2d(80, 128, 1)
        self.lin_8 = nn.Conv2d(160, 256, 1)

    def forward(self, data):
        B, _, H, W = data['image0'].shape
        x = torch.cat([data['image0'], data['image1']], 0)
        feature_pyramid = self.cnn.forward_features_8(x)
        feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)
        feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)

        scale = 8
        h_8, w_8 = H // scale, W // scale
        device = data['image0'].device
        grid = [rearrange((create_meshgrid(h_8, w_8, False, device) * scale).squeeze(0),
                          'h w t->(h w) t')] * B  # kpt_xy
        grid_8 = torch.stack(grid, 0)

        data.update({
            'bs': B,
            'c': feat_8_0.shape[1],
            'h_8': h_8,
            'w_8': w_8,
            'hw_8': h_8 * w_8,
            'feat_8_0': feat_8_0,
            'feat_8_1': feat_8_1,
            'feat_4_0': feat_4_0,
            'feat_4_1': feat_4_1,
            'grid_8': grid_8,
        })






import torch.nn as nn
import torch.nn.functional as F
from .repvgg import create_RepVGG

class RepVGG_8_1_align(nn.Module):
    """
    RepVGG backbone, output resolution are 1/8 and 1.
    Each block has 2 layers.
    """

    def __init__(self, config=None):
        super().__init__()
        backbone = create_RepVGG(False)

        self.layer0, self.layer1, self.layer2, self.layer3 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3

        for layer in [self.layer0, self.layer1, self.layer2, self.layer3]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        out = self.layer0(x) # 1/2 64
        for module in self.layer1:
            out = module(out) # 1/2 64
        x1 = out
        for module in self.layer2:
            out = module(out) # 1/4  128
        x2 = out
        for module in self.layer3:
            out = module(out) # 1/8  258
        x3 = out
        return {'feats_c': x3, 'feats_x2': x2, 'feats_x1': x1}
        # feat_8_0, feat_8_1 = x3.split(B)
        # feat_4_0, feat_4_1 = x2.split(B)
        # feat_2_0, feat_2_1 = x1.split(B)
        #
        #
        # scale = 8
        # h_8, w_8 = H // scale, W // scale
        # device = data['image0'].device
        # grid = [rearrange((create_meshgrid(h_8, w_8, False, device) * scale).squeeze(0),
        #                   'h w t->(h w) t')] * B  # kpt_xy
        # grid_8 = torch.stack(grid, 0)
        #
        # data.update({
        #     'bs': B,
        #     'c': feat_8_0.shape[1],
        #     'h_8': h_8,
        #     'w_8': w_8,
        #     'hw_8': h_8 * w_8,
        #     'feat_8_0': feat_8_0,
        #     'feat_8_1': feat_8_1,
        #     'feat_4_0': feat_4_0,
        #     'feat_4_1': feat_4_1,
        #     'feat_2_0': feat_2_0,
        #     'feat_2_1': feat_2_1,
        #     'grid_8': grid_8,
        # })
