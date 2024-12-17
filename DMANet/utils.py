""" utils.py
"""

import os
import random

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import warnings
import scipy
import math

name_dataparallel = torch.nn.DataParallel.__name__
log10 = np.log(10)


def set_seed(seed):

    # 内置模块的种子也要设
    random.seed(seed)

    # 设置Numpy的种子
    np.random.seed(seed)

    # 设置PyTorch的种子
    torch.manual_seed(seed)

    # 如果使用GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU

    # 确保完全可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BinaryMaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, low_threshold=0.4, high_threshold=0.6):
        # 保存上下文以备反向传播使用
        ctx.save_for_backward(mask)
        ctx.low_threshold = low_threshold
        ctx.high_threshold = high_threshold

        # 在前向传播中进行二值化操作
        binary_mask = torch.zeros_like(mask)
        binary_mask[mask > high_threshold] = 1
        binary_mask[mask < low_threshold] = 0
        binary_mask[(mask >= low_threshold) & (mask <= high_threshold)] = mask[
            (mask >= low_threshold) & (mask <= high_threshold)]

        return binary_mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        low_threshold = ctx.low_threshold
        high_threshold = ctx.high_threshold

        grad_mask = grad_output.clone()
        # 保持梯度链
        grad_mask[mask > high_threshold] = grad_output[mask > high_threshold]
        grad_mask[mask < low_threshold] = grad_output[mask < low_threshold]
        grad_mask[(mask >= low_threshold) & (mask <= high_threshold)] = grad_output[
            (mask >= low_threshold) & (mask <= high_threshold)]

        return grad_mask


def fusion_image(b256, b2562, mask, low, high):
    binary_mask = BinaryMaskFunction.apply(mask)

    # 创建掩码
    mask_A = (binary_mask > high).float()
    mask_B = (binary_mask < low).float()

    weighted_region = mask * b256 + (1 - mask) * b2562

    # 创建结果图像，使用 torch.where 来确保梯度流动
    result = torch.where(mask_A == 1, b256, torch.where(mask_B == 1, b2562, weighted_region))

    return result


def fusion_image_deblur(b256, b2562, db256, db2562, mask, low, high):
    binary_mask = BinaryMaskFunction.apply(mask, low, high)

    # 创建掩码
    mask_A = (binary_mask > high).float()
    mask_B = (binary_mask < low).float()

    weighted_region = mask * b256 + (1 - mask) * b2562
    # todo 用deblur图来加权一下，前面这样做好像和mask没关系

    # 创建结果图像，使用 torch.where 来确保梯度流动
    result = torch.where(mask_A == 1, b256, torch.where(mask_B == 1, b2562, weighted_region))

    return result


def xavier_init_normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def compute_psnr(x, label, max_diff):
    assert max_diff in [255, 1, 2]
    if max_diff == 255:
        x = x.clamp(0, 255)
    elif max_diff == 1:
        x = x.clamp(0, 1)
    elif max_diff == 2:
        x = x.clamp(-1, 1)

    mse = ((x - label) ** 2).mean()
    return 10 * torch.log(max_diff ** 2 / mse) / log10


def lr_warmup(epoch, warmup_length):
    if epoch < warmup_length:
        p = max(0.0, float(epoch)) / float(warmup_length)
        p = 1.0 - p
        return np.exp(-p * p * 5.0)
    else:
        return 1.0


def load_optimizer(optimizer, model, path, epoch=None):
    """
    return the epoch
    """
    if type(model).__name__ == name_dataparallel:
        model = model.module

    if epoch is None:
        for i in reversed(range(10000)):
            p = "{}/{}_epoch{}.pth".format(path, type(optimizer).__name__ + '_' + type(model).__name__, i)
            if os.path.exists(p):
                optimizer.load_state_dict(torch.load(p))
                return i
    else:
        p = "{}/{}_epoch{}.pth".format(path, type(optimizer).__name__ + '_' + type(model).__name__, epoch)
        if os.path.exists(p):
            optimizer.load_state_dict(torch.load(p))
            return epoch
        else:
            warnings.warn("resume optimizer not found at {}".format(p))

    warnings.warn("resume model not found ")
    return -1


def load_model(model, path, epoch=None, strict=True):
    """
    return the last epoch
    """
    if type(model).__name__ == name_dataparallel:
        model = model.module
    if epoch is None:
        for i in reversed(range(10000)):
            p = "{}/{}_epoch{}.pth".format(path, type(model).__name__, i)
            if os.path.exists(p):
                print(p)
                model.load_state_dict(torch.load(p), strict=strict)
                return i
    else:
        p = "{}/{}_epoch{}.pth".format(path, type(model).__name__, epoch)
        if os.path.exists(p):
            checkpoint = torch.load(p)
            print(checkpoint.keys())
            print(p)
            model.load_state_dict(torch.load(p), strict=strict)
            return epoch
        else:
            warnings.warn("resume model not found at {}".format(p))

    warnings.warn("resume model not found ")
    return -1


def set_requires_grad(module, b):
    for parm in module.parameters():
        parm.requires_grad = b


def adjust_dyn_range(x, drange_in, drange_out):
    if not drange_in == drange_out:
        scale = float(drange_out[1] - drange_out[0]) / float(drange_in[1] - drange_in[0])
        bias = drange_out[0] - drange_in[0] * scale
        x = x.mul(scale).add(bias)
    return x


def resize(x, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(size),
        transforms.ToTensor(),
    ])
    return transform(x)


def make_image_grid(x, ngrid):
    x = x.clone().cpu()
    if pow(ngrid, 2) < x.size(0):
        grid = make_grid(x[:ngrid * ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid * ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
    return grid


def save_image_single(x, path, imsize=512):
    from PIL import Image
    grid = make_image_grid(x, 1)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize, imsize), Image.NEAREST)
    im.save(path)


def save_image_grid(x, path, imsize=512, ngrid=4):
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize, imsize), Image.NEAREST)
    im.save(path)


def save_model(model, dirname, epoch, best=''):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    torch.save(model.state_dict(), '{}/{}_epoch{}.pth'.format(dirname, type(model).__name__, epoch))


def save_optimizer(optimizer, model, dirname, epoch):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    torch.save(optimizer.state_dict(),
               '{}/{}_epoch{}.pth'.format(dirname, type(optimizer).__name__ + '_' + type(model).__name__, epoch))



irange = range

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid
