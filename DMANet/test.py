import torch
import os
import cv2
from network import Netm,GKMNetFixedAPUNoC2FProgressiveUNet
from data import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model, set_requires_grad
from time import time
import test_config as config


if __name__ == '__main__':
    dataset = TestDataset(config.train['test_img_path'], config.train['test_gt_path'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                            pin_memory=True)

    net = GKMNetFixedAPUNoC2FProgressiveUNet(kernel_mode=config.train['kernel_mode'], num_res=2, base_ch=config.train['base_ch'], num_gaussian_kernels=config.train['kernel_size'],
                                                                   gaussian_kernel_size=config.train['kernel_size']).cuda()

    netm = Netm().cuda()

    set_requires_grad(net, False)
    set_requires_grad(netm, False)

    last_epoch = load_model(net, config.train['resume'], epoch=config.train['resume_epoch'])
    last_epochm = load_model(netm, config.train['resume'], epoch=config.train['resume_epoch'])

    log_dir = 'test/{}'.format('DPD')
    os.system('mkdir -p {}'.format(log_dir))

    total_time = 0
    net.eval()
    with torch.no_grad():
        for step, (batch,name,pad_w,pad_h) in tqdm(enumerate(dataloader), total=len(dataloader)):
            name = name[0]
            tt = time()
            for k in batch:
                batch[k] = batch[k].cuda(non_blocking=True)
                batch[k].requires_grad = False


            db256, db128, db64,db2562, db1282, db642, t,weights,weights2 = net(batch['img256'], batch['img2562'], 0)


            if config.train['netm_use_image'] == 'blur':
                out_m, out_m2 = netm(batch['img256'], batch['img2562'], weights, weights2)
            elif config.train['netm_use_image'] == 'deblur':
                out_m, out_m2 = netm(db256, db2562, weights, weights2)
            else:
                out_m, out_m2 = netm(db256, db2562, batch['img256'], batch['img2562'], weights, weights2)

            _, _, h, w = db256.shape

            resultdb256 = db256[:, :, int(np.floor(pad_h/2)):h - int(np.ceil(pad_h/2)), int(np.floor(pad_w/2)):w - int(np.ceil(pad_w/2))]

            output_dir = config.train['output_dir']

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            cv2.imwrite(output_dir + str(name).split('.')[0] + '_x.png',
                        cv2.cvtColor(resultdb256[0].cpu().numpy().transpose([1, 2, 0]) * 255., cv2.COLOR_BGR2RGB))

            resultdb2562 = db2562[:, :, int(np.floor(pad_h / 2)):h - int(np.ceil(pad_h / 2)),
                          int(np.floor(pad_w / 2)):w - int(np.ceil(pad_w / 2))]
            cv2.imwrite(output_dir + str(name).split('.')[0] + '_x1.png',
                        cv2.cvtColor(resultdb2562[0].cpu().numpy().transpose([1, 2, 0]) * 255., cv2.COLOR_BGR2RGB))

            resultout_m = out_m[:, :, int(np.floor(pad_h / 2)):h - int(np.ceil(pad_h / 2)),
                            int(np.floor(pad_w / 2)):w - int(np.ceil(pad_w / 2))]
            cv2.imwrite(output_dir + str(name).split('.')[0] + '_m0.png',
                        cv2.cvtColor(resultout_m[0].cpu().numpy().transpose([1, 2, 0]) * 255., cv2.COLOR_BGR2RGB))

            resultout_m2 = out_m2[:, :, int(np.floor(pad_h / 2)):h - int(np.ceil(pad_h / 2)),
                          int(np.floor(pad_w / 2)):w - int(np.ceil(pad_w / 2))]
            cv2.imwrite(output_dir + str(name).split('.')[0] + '_m1.png',
                        cv2.cvtColor(resultout_m2[0].cpu().numpy().transpose([1, 2, 0]) * 255., cv2.COLOR_BGR2RGB))

