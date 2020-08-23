# config.py

cfg_mnetv1 = {
    'name': 'mobilenet0.25',
    'min_sizes': [[64, 128], [256, 512]],
    'steps': [16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 1,
    'epoch': 120,
    'decay1': 80,
    'decay2': 100,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_mnetv2 = {
    'name': 'mobilenetv2_0.1',
    'min_sizes': [[64, 128], [256, 512]],
    'steps': [16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 1,
    'epoch': 120,
    'decay1': 80,
    'decay2': 100,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage2': 2, 'stage3': 3},
    'in_channel1': 12,
    'in_channel2': 1280,
    'out_channel': 64
}

cfg_mnetv3 = {
    'name': 'mobilenetv3',
    'min_sizes': [[64, 128], [256, 512]],
    'steps': [16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 120,
    'decay1': 80,
    'decay2': 100,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage2': 2, 'stage3': 3},
    'in_channel1': 48,
    'in_channel2': 576,
    'out_channel': 64
}

cfg_efnetb0 = {
    'name': 'efficientnetb0',
    'min_sizes': [[64, 128], [256, 512]],
    'steps': [16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 1,
    'epoch': 120,
    'decay1': 80,
    'decay2': 100,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage2': 2, 'stage3': 3},
    'in_channel1': 112,
    'in_channel2': 1280,
    'out_channel': 64
}



