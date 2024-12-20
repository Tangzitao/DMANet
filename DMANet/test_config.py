# test_config
train = {}
train['test_img_path'] = 'D:\DLproj\MFIF\dataset/MFI-WHU/'
train['test_gt_path'] = 'D:\DLproj\MFIF\dataset/MFI-WHU/'

train['test_batch_size'] = 1
train['beta'] = 2
train['kernel_size'] = 15
train['kernel_mode'] = 'FG'
train['base_ch'] = 32
train['netm_use_image'] = 'deblur'  # blur or deblur


# config for save , log and resume
train['retrain'] = False
train['resume'] = './checkpoints/'
train['resume_epoch'] = 39
train['output_dir'] = './our_result/WHU_result/1/'

