import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import einops


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True,
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          padding_mode='reflect'))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state
        combined = torch.cat((input, hidden), 1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())


class res_block(nn.Module):
    def __init__(self, ch_in):
        super(res_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True))

    def forward(self, x):
        y = x + self.conv(x)
        return y + self.conv1(y)




class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class SqueezeAttentionBlockNoAct(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlockNoAct, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = BasicConv(ch_in, ch_out, 1, 1, relu=False)
        self.conv_atten = CLSTM_cell(ch_in, ch_out, 5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.out=BasicConv(ch_out, ch_out, 3, 1)

    def forward(self, x, hidden_state):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        h, c = self.conv_atten(y, hidden_state)
        y = self.upsample(h)
        y=self.out(y)+x_res
        return y, h, c
        # return (y * x_res) + y, h, c


def create_position_feats(shape, scales=None, bs=1, device=None):
    cord_range = [range(s) for s in shape]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)
    mesh = torch.from_numpy(mesh)
    if device is not None:
        mesh = mesh.to(device)
    if scales is not None:
        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales, dtype=torch.float32, device=device)
        mesh = mesh * (1.0 / scales.view(-1, 1, 1))
    return torch.stack(bs * [mesh])



def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return (gauss / gauss.sum()).cuda()


def gen_gaussian_kernel(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
    return window


def generate_1d_kernel(size, beta):
    kernel = torch.zeros(size)
    center_index = size // 2
    kernel[center_index] = 1.0  # 中间值
    value = 1.0
    for i in range(1, center_index + 1):
        value /= beta
        kernel[center_index - i] = value
        kernel[center_index + i] = value
    kernel /= kernel.sum()  # 归一化，使卷积核的值和为1
    return kernel


def generate_2d_kernel(size, beta):
    _1D_window = generate_1d_kernel(size, beta).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(1, 1, size, size).contiguous())
    return window


def shape2polar_coordinate(shape=(3,3), device='cuda'):
    h, w=shape
    x=torch.arange(0, h, device=device)
    y=torch.arange(0, w, device=device)

    x, y=torch.meshgrid(x, y)

    min=-1
    max=1
    x=x/(h-1)*(max-min)+min
    y=y/(w-1)*(max-min)+min
    cord=x+1j*y


    r=torch.abs(cord)/np.sqrt(2)
    theta=torch.angle(cord)
    theta_code=torch.cat([(torch.cos(theta).unsqueeze(-1)+1)/2, (torch.sin(theta).unsqueeze(-1)+1)/2], dim=-1)

    cord=torch.cat([r.unsqueeze(-1), theta_code], dim=-1)

    return cord


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class KernelINR_Polar(nn.Module):
    def __init__(self, hidden_dim=64, w=1.):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(3, hidden_dim),
            Sine(w),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(w),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, cord):
        k=self.layers(cord).squeeze(-1)
        return k

class SizeGroupINRConvPolar(nn.Module):
    def __init__(self, max_kernel_size=17, num_ch=3, basis_num=5, w_max=7., w_min=1., w_list=None, learnable_freq=False):
        super(SizeGroupINRConvPolar, self).__init__()
        if w_list is None:
            w_list=[w_min+(w_max-w_min)/(basis_num-1)*i for i in range(basis_num)]
            if learnable_freq:
                newwlist=[torch.nn.Parameter(torch.scalar_tensor(w_list[i], dtype=torch.float32)) for i in range(basis_num)]
                w_list=newwlist
        assert len(w_list)==basis_num
        self.w_list=w_list
        self.num_ch=num_ch
        self.kernelINR_list=nn.ModuleList(KernelINR_Polar(hidden_dim=64, w=w_list[i]) for i in range(basis_num))

        self.basis_num=basis_num
        self.max_kernel_size=max_kernel_size
        self.kernel_sizes=[(2*(i+1)+1, 2*(i+1)+1) for i in range(max_kernel_size//2)]
        # self.kernel_sizes = [(max_kernel_size, max_kernel_size) for i in range(max_kernel_size // 2)]

        self.padding=max_kernel_size//2
        self.group_num=len(self.kernel_sizes)

        masks=[] # [1x1, 3x3, ..., 15x15, ...]

        cords=[] # [1x1xc, 3x3xc, ..., 15x15xc, ...]

        empty=torch.zeros(self.basis_num, 1, 1, 1, device='cuda')
        # delta[0, :,max_kernel_size//2, max_kernel_size//2]=1
        delta = torch.ones(self.basis_num, 1, 1, 1, device='cuda')

        self.delta=delta
        self.empty=empty

        for siz in self.kernel_sizes:
            mask = torch.ones(siz, device='cuda', dtype=torch.float32) * (3 ** 2) / (siz[0] * siz[1])

            masks.append(mask)

            cord=shape2polar_coordinate(shape=siz, device='cuda')
            cords.append(cord)

        self.masks=masks
        self.cords=cords


    def forward(self, x):
        b, c, h, w = x.shape
        kernels = []
        maps = []
        for k in range(self.group_num) :
            kernels_g = []
            for i in range(self.basis_num):
                kernel = self.kernelINR_list[i](self.cords[k])  # h w
                kernel = kernel*self.masks[k]
                kernels_g.append(kernel.unsqueeze(0))
            kernels_g=torch.cat(kernels_g, dim=0)  # m h w
            maps_g = F.conv2d(x, kernels_g.repeat(self.num_ch, 1, 1).unsqueeze(1),
                              padding=self.kernel_sizes[k][0]//2,
                              groups=self.num_ch)  # b 3*m h w
            maps.append(maps_g.unsqueeze(1))
            kernels.append(kernels_g)
        maps=torch.cat(maps, dim=1)  # b gn 3*m h w

        null_map=torch.zeros(b, 1, self.num_ch*self.basis_num, h, w, device='cuda')

        maps=torch.cat([null_map, maps], dim=1)  # b gn+1 3*m h w

        return maps

class GaussianBlurLayer(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='TG', channels=3):
        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        kernel_size = 3
        weight = torch.zeros(num_kernels + 1, 1, max_kernel_size, max_kernel_size)  # 最后会有一个全0，ok
        for i in range(num_kernels):
            pad = int((max_kernel_size - kernel_size) / 2)
            weight[i + 1] = (F.pad(gen_gaussian_kernel(kernel_size, sigma=0.25 * (i + 1)).cuda(),
                                   [pad, pad, pad, pad])).squeeze(0)
            if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
                kernel_size += 2
        pad = int((max_kernel_size - 1) / 2)
        weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
                           [pad, pad, pad, pad])).squeeze(0)

        # kernel=weight
        kernel = torch.tile(weight, dims=(3,1,1,1)).cuda()  # 将张量沿着第一个维度复制三次，对应rgb通道 1,2,3,..,16,1,..,16,1,2,..,16
        kernel = nn.Parameter(data=kernel)
        if mode == 'TG':  # Trainable Gaussion
            self.weight = kernel
            self.weight.requires_grad = True

        elif mode == 'TR':  # 这里是可训练的，但随机初始化
            self.weight = nn.Parameter(data=torch.randn((num_kernels+1) * 3, 1, max_kernel_size, max_kernel_size),
                                       requires_grad=True)

        elif mode == 'TSG':
            kernel_size = 3
            weight = torch.zeros(num_kernels + 1, 1, max_kernel_size, max_kernel_size)
            initial_sigmas = torch.tensor([0.25 * (i + 1) for i in range(num_kernels)], dtype=torch.float32)
            self.sigma_list = nn.Parameter(initial_sigmas, requires_grad=True)
            for i in range(num_kernels):
                pad = int((max_kernel_size - kernel_size) / 2)
                weight[i + 1] = (F.pad(gen_gaussian_kernel(kernel_size, sigma=self.sigma_list[i]).cuda(),
                                       [pad, pad, pad, pad])).squeeze(0)
                if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
                    kernel_size += 2
            pad = int((max_kernel_size - 1) / 2)
            weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
                               [pad, pad, pad, pad])).squeeze(0)
            kernel = torch.tile(weight, dims=(3, 1, 1, 1)).cuda()
            self.weight = kernel

        elif mode == 'FD':
            kernel_size = 3
            weight = torch.zeros(num_kernels + 1, 1, max_kernel_size, max_kernel_size)
            for i in range(num_kernels):
                pad = int((max_kernel_size - kernel_size) / 2)
                beta = 2 if i % 2 == 0 else 3
                weight[i + 1] = (F.pad(generate_2d_kernel(kernel_size, beta).cuda(),
                                       [pad, pad, pad, pad])).squeeze(0)
                if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
                    kernel_size += 2
            pad = int((max_kernel_size - 1) / 2)
            weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
                               [pad, pad, pad, pad])).squeeze(0)
            kernel = torch.tile(weight, dims=(3, 1, 1, 1)).cuda()
            self.weight = kernel

        else:  # Fixed Gaussion
            self.weight = kernel
            self.weight.requires_grad = False
        self.padding = int((max_kernel_size - 1) / 2)

    def __call__(self, x):

        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)  # 分组之后得到的卷积也是1,..,16,1,..,16,...前16个都是与第一通道的计算结果
        return x, self.weight


