from layers import *
import torch
from time import time
import torch.nn.functional as F
from skip import skip


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, num_res=2):
        super(UNet, self).__init__()
        self.Encoder = nn.ModuleList([
            EBlock(base_ch, num_res),
            EBlock(base_ch * 2, num_res),
            EBlock(base_ch * 4, num_res),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_ch*2, num_res),
            DBlock(base_ch, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_ch, base_ch, kernel_size=3, relu=True, stride=1),
            BasicConv(base_ch * 1, base_ch * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 2, base_ch * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 4, base_ch * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch * 1, kernel_size=3, relu=True, stride=1)
        ])

        self.up1=BasicConv(base_ch * 4, base_ch * 2, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up2=BasicConv(base_ch * 2, base_ch * 1, kernel_size=4, relu=True, stride=2, transpose=True)



    def forward(self, x):
        '''Feature Extract 0'''
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        '''Down Sample 1'''
        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        '''Down Sample 2'''
        z = self.feat_extract[2](res2)
        res3 = self.Encoder[2](z)

        deepz=res3

        '''Up Sample 2'''
        z=self.up1(res3)
        z = self.feat_extract[3](torch.cat([z, res2], dim=1))
        z = self.Decoder[0](z)

        '''Up Sample 1'''
        z=self.up2(z)
        z = self.feat_extract[4](torch.cat([z, res1], dim=1))
        z = self.Decoder[1](z)

        return z, deepz


class ProgressiveUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, num_res=2):
        super(ProgressiveUNet, self).__init__()
        self.Encoder = nn.ModuleList([
            EBlock(base_ch, num_res),
            EBlock(base_ch * 2, num_res),
        ])

        self.progressive_blocks = nn.ModuleList([
            EBlock(base_ch * 4, num_res),
            EBlock(base_ch * 4, num_res),
            EBlock(base_ch * 4, num_res)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_ch * 2, num_res),
            DBlock(base_ch, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_ch, base_ch, kernel_size=3, relu=True, stride=1),
            BasicConv(base_ch * 1, base_ch * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 2, base_ch * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 4, base_ch * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch * 1, kernel_size=3, relu=True, stride=1)
        ])

        self.up1 = BasicConv(base_ch * 4, base_ch * 2, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up2 = BasicConv(base_ch * 2, base_ch * 1, kernel_size=4, relu=True, stride=2, transpose=True)

    def forward(self, x, scale=1):
        '''Feature Extract 0'''
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        '''Down Sample 1'''
        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        '''Down Sample 2'''
        z = self.feat_extract[2](res2)

        for i in range(0, scale):
            z = self.progressive_blocks[i](z)

        res3 = z

        deepz = res3

        '''Up Sample 2'''
        z = self.up1(res3)
        z = self.feat_extract[3](torch.cat([z, res2], dim=1))
        z = self.Decoder[0](z)

        '''Up Sample 1'''
        z = self.up2(z)
        z = self.feat_extract[4](torch.cat([z, res1], dim=1))
        z = self.Decoder[1](z)

        return z, deepz


class DMANet(nn.Module):
    def __init__(self, kernel_mode='FG', num_res=2, base_ch=32, num_gaussian_kernels=21, gaussian_kernel_size=21):
        super(type(self), self).__init__()
        super().__init__()
        self.GCM = GaussianBlurLayer(num_gaussian_kernels, gaussian_kernel_size, kernel_mode, channels=48)

        '''backbone'''
        self.unet=ProgressiveUNet(3, base_ch, num_res=num_res)
        '''APU'''
        self.APU = SqueezeAttentionBlockNoAct(base_ch, base_ch)
        self.beta=nn.Conv2d(base_ch, (num_gaussian_kernels+1) * 3, kernel_size=1)
        '''summation'''
        self.SumLayer = nn.Conv2d((num_gaussian_kernels+1) * 3, 3, kernel_size=1, bias=False)

    # weight_first
    def forward_step(self, input_blurry, hidden_state, scale=1):

        blur_input = input_blurry

        f, _ = self.unet(blur_input, scale)

        weights, h, c = self.APU(f, hidden_state)
        weights = self.beta(weights)

        blurry_repeat = torch.repeat_interleave(input_blurry, repeats=16, dim=1)
        gt_x = blurry_repeat*weights
        gt_y, kernels = self.GCM(gt_x)

        '''Summation'''
        result = self.SumLayer(gt_y)
        result = input_blurry + result

        return result, h, c, weights

    def forward(self, x, x2, gt):
        input_blur_256=x
        input_blur_128=F.interpolate(x, scale_factor=0.5, mode='bilinear')
        input_blur_64=F.interpolate(x, scale_factor=0.25, mode='bilinear')
        input_blur_2562=x2
        input_blur_1282=F.interpolate(x2, scale_factor=0.5, mode='bilinear')
        input_blur_642=F.interpolate(x2, scale_factor=0.25, mode='bilinear')
        gt_128, gt_64 = 0, 0
        h, c = self.APU.conv_atten.init_hidden(
            input_blur_64.shape[0],
            (input_blur_64.shape[-2] // 2, input_blur_64.shape[-1] // 2))
        h2, c2 = self.APU.conv_atten.init_hidden(
            input_blur_642.shape[0],
            (input_blur_642.shape[-2] // 2, input_blur_642.shape[-1] // 2))
        betas=[]
        """The forward process"""
        '''scale 1'''
        db64, h, c, beta = self.forward_step(input_blur_64, (h, c), scale=1)
        db642, h2, c2, beta2 = self.forward_step(input_blur_642, (h2, c2), scale=1)
        betas.append(beta)
        h = F.upsample(h, scale_factor=2, mode='bilinear')
        c = F.upsample(c, scale_factor=2, mode='bilinear')
        betas.append(beta2)
        h2 = F.upsample(h2, scale_factor=2, mode='bilinear')
        c2 = F.upsample(c2, scale_factor=2, mode='bilinear')
        '''scale 2'''
        db128, h, c, beta = self.forward_step(input_blur_128, (h, c), scale=2)
        betas.append(beta)
        h = F.upsample(h, scale_factor=2, mode='bilinear')
        c = F.upsample(c, scale_factor=2, mode='bilinear')
        db1282, h2, c2, beta2 = self.forward_step(input_blur_1282,  (h2, c2), scale=2)
        betas.append(beta2)
        h2 = F.upsample(h2, scale_factor=2, mode='bilinear')
        c2 = F.upsample(c2, scale_factor=2, mode='bilinear')
        '''scale 3'''
        db256, _, _, beta = self.forward_step(input_blur_256,  (h, c), scale=3)
        betas.append(beta)
        db2562, _, _, beta2 = self.forward_step(input_blur_2562,  (h2, c2), scale=3)
        betas.append(beta2)
        return db256, db128, db64,db2562, db1282, db642, time(),beta,beta2



class Netm(nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        super().__init__()
        self.netm1 = skip(102, 1,
                             channels=[128, 128, 128],
                             channels_skip=16,
                             upsample_mode='bilinear',
                             need_bias=False, pad='reflection', act_fun='LeakyReLU', scales=3)

    def forward(self, db256, db2562, weights, weights2):
        input_m = torch.cat([db256, db2562, weights, weights2], dim=1)
        out_m = self.netm1(input_m)

        input_m2 = torch.cat([db2562, db256, weights2, weights], dim=1)
        out_m2 = self.netm1(input_m2)

        return out_m, out_m2
