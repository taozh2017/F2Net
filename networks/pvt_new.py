import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.pvtv2_new import pvt_v2_b2
from networks.pvtv2_new import pvt_v2_encoder
from einops import rearrange


class Doubleconv(nn.Module):
    def __init__(self, in_chan, out_chan, mid_channels=None):
        super(Doubleconv, self).__init__()
        if mid_channels is None:
            mid_channels = out_chan
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_chan, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_chan)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class Up(nn.Module):
    def __init__(self, in_chan, out_chan, bilinear=True):
        super(Up, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Doubleconv(in_chan + out_chan, out_chan, out_chan)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=2, stride=2)
            self.conv = Doubleconv(out_chan * 2, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class Fusion_layer(nn.Module):
    def __init__(self, dim):
        super(Fusion_layer, self).__init__()
        self.dim = dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        x1_0 = x1 * x2
        x1_0 = torch.cat((x1_0, x1), dim=1)
        x1_0 = self.conv1(x1_0)

        x2_0 = x1 * x3
        x2_0 = torch.cat((x2_0, x1), dim=1)
        x2_0 = self.conv2(x2_0)

        x = torch.cat((x1_0, x2_0), dim=1)
        x = self.conv3(x)

        x3_0 = x1 * x2 * x3
        x = torch.cat((x3_0, x), dim=1)
        x = self.conv4(x)

        return x


class Final_Up(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(Final_Up, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim * dim_scale ** 2, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C)
        x = self.expand(x)
        x = x.view(B, H, W, -1)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class F2Net(nn.Module):
    def __init__(self, num_classes):
        super(F2Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
        self.backbone0 = pvt_v2_encoder()  # [64, 128, 320, 512]
        self.backbone1 = pvt_v2_b2()  # [64, 128, 320, 512]
        self.backbone2 = pvt_v2_b2()  # [64, 128, 320, 512]

        path = './networks/pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone1.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone0.load_state_dict(model_dict, False)
        self.backbone1.load_state_dict(model_dict)
        self.backbone2.load_state_dict(model_dict)

        self.up11 = Up(512, 320, False)
        self.up12 = Up(320, 128, False)
        self.up13 = Up(128, 64, False)
        self.up21 = Up(512, 320, False)
        self.up22 = Up(320, 128, False)
        self.up23 = Up(128, 64, False)

        self.up01 = Up(512, 320, False)
        self.up02 = Up(320, 128, False)
        self.up03 = Up(128, 64, False)
        self.fusion_layer1 = Fusion_layer(320)
        self.fusion_layer2 = Fusion_layer(128)
        self.fusion_layer3 = Fusion_layer(64)
        self.up0 = Final_Up((4, 4), 64, 4)
        self.up1 = Final_Up((4, 4), 64, 4)
        self.up2 = Final_Up((4, 4), 64, 4)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
        )
        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
        )

    def forward(self, x):
        # backbone
        x1 = torch.cat((x[:, 1, None, :, :], x[:, 2, None, :, :]), dim=1)
        x2 = torch.cat((x[:, 0, None, :, :], x[:, 3, None, :, :]), dim=1)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        pvt1 = self.backbone1(x1)
        pvt2 = self.backbone2(x2)

        # -- stage encoder fusion
        x = self.backbone0(pvt1, pvt2)

        # -- stage decoder
        x1 = self.up11(pvt1[3], pvt1[2])
        x1 = self.up12(x1, pvt1[1])
        x1 = self.up13(x1, pvt1[0])

        x2 = self.up21(pvt2[3], pvt2[2])
        x2 = self.up22(x2, pvt2[1])
        x2 = self.up23(x2, pvt2[0])

        # -- stage decoder fusion

        x0 = self.fusion_layer1(x[2], pvt1[2], pvt2[2])
        logits = self.up01(x[3], x0)
        x0 = self.fusion_layer2(x[1], pvt1[1], pvt2[1])
        logits = self.up02(logits, x0)
        x0 = self.fusion_layer3(x[0], pvt1[0], pvt2[0])
        logits = self.up03(logits, x0)

        logits = self.up0(logits)
        logits = self.out(logits)

        x1 = self.up1(x1)
        x1 = self.out1(x1)

        x2 = self.up2(x2)
        x2 = self.out2(x2)

        return logits, x1, x2
