import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.ops.functional as F

# from mindcv.models.pvtv2 import pvt_v2_b2
from mindcv.models.pvtv2_new import pvt_v2_b2, pvt_v2_b2_encoder


class Doubleconv(nn.Cell):
    def __init__(self, in_chan, out_chan, mid_channels=None):
        super(Doubleconv, self).__init__()
        if mid_channels is None:
            mid_channels = out_chan
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=mid_channels, kernel_size=3, stride=1, pad_mode='same',
                               has_bias=False)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_chan, kernel_size=3, stride=1,
                               pad_mode='same', has_bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_chan)

    def construct(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class Up(nn.Cell):
    """
    Upsampling high_feature with factor=2 and concat with low feature
    """

    def __init__(self, in_chan, out_chan, bilinear=True):
        super(Up, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.concat = P.Concat(axis=1)
        if not bilinear:
            self.up = nn.Conv2dTranspose(in_chan, out_chan, kernel_size=2, stride=2, pad_mode="same")
            self.conv = Doubleconv(out_chan * 2, out_chan)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x = self.concat((x2, x1))
        x = self.conv(x)
        return x
    

class Final_Up(nn.Cell):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(Final_Up, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Dense(dim, dim * dim_scale ** 2, has_bias=False)
        self.output_dim = dim
        self.norm = norm_layer([self.output_dim])

    def construct(self, x):
        B, C, H, W = x.shape
        x = P.transpose(x, (0, 2, 3, 1))
        x = P.reshape(x, (B, -1, C))
        x = self.expand(x)
        x = P.reshape(x, (B, H, W, -1))

        x = P.reshape(x, (B, H * self.dim_scale, W * self.dim_scale, -1))
        x = self.norm(x)
        x = P.transpose(x, (0, 3, 1, 2))
        return x
    

class Fusion_layer(nn.Cell):
    def __init__(self, dim):
        super(Fusion_layer, self).__init__()
        self.dim = dim
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(dim * 2, dim, kernel_size=3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(dim * 2, dim, kernel_size=3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(dim * 2, dim, kernel_size=3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.conv4 = nn.SequentialCell(
            nn.Conv2d(dim * 2, dim, kernel_size=3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.concat = P.Concat(axis=1)

    def construct(self, x1, x2, x3):
        x1_0 = x1 * x2
        x1_0 = self.concat((x1_0, x1))
        x1_0 = self.conv1(x1_0)

        x2_0 = x1 * x3
        x2_0 = self.concat((x2_0, x1))
        x2_0 = self.conv2(x2_0)

        x = self.concat((x1_0, x2_0))
        x = self.conv3(x)

        x3_0 = x1 * x2 * x3
        x = self.concat((x3_0, x))
        x = self.conv4(x)

        return x


class F2Net(nn.Cell):
    def __init__(self, args):
        super(F2Net, self).__init__()
        self.args = args
        
        self.backbone0 = pvt_v2_b2_encoder(
            in_channels=3,
            drop_rate=0.0,
            drop_path_rate=0.1,
        )

        self.backbone1 = pvt_v2_b2(
            in_channels=3,
            drop_rate=0.0,
            drop_path_rate=0.1,
        )

        self.backbone2 = pvt_v2_b2(
            in_channels=3,
            drop_rate=0.0,
            drop_path_rate=0.1,
        )
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, pad_mode="same", has_bias=True)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, pad_mode="same", has_bias=True)

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
        
        self.out = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=args.num_classes, kernel_size=1, has_bias=True)
        )
        self.out1 = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=args.num_classes, kernel_size=1, has_bias=True)
        )
        self.out2 = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=args.num_classes, kernel_size=1, has_bias=True)
        )

        self.concat = P.Concat(axis=1)

    def construct(self, x):
        
        x1 = self.concat((x[:, 1, None, :, :], x[:, 2, None, :, :]))
        x2 = self.concat((x[:, 0, None, :, :], x[:, 3, None, :, :]))
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        
        # backbone
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