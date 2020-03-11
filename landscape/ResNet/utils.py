from math import pi, cos
import numpy as np
import os
import shutil
import torch

def load_state_ckpt(model_path, model):
    checkpoint = torch.load(model_path)

    model_dict = model.state_dict()

    for key, v in checkpoint['state_dict'].items():
        if key in model_dict:
            #print(key)
            v1 = model_dict[key]
            if len(v.shape) != len(v1.shape):
                assert v1.shape[:2] == v.shape[:2], \
                     ('Workspace blob {} with shape {} does not match '
                     'weights file shape {}').format( key, v1.shape, v.shape)

                assert v1.shape[-2:] == v.shape[-2:], \
                     ('Workspace blob {} with shape {} does not match '
                      'weights file shape {}').format(key, v1.shape, v.shape)

                num_inflate = v1.shape[2]
                checkpoint['state_dict'][key] = torch.stack([checkpoint['state_dict'][key]] * num_inflate,dim=2) / float(num_inflate)

                assert v1.shape == checkpoint['state_dict'][key].shape, \
                     ('Workspace blob {} with shape {} does not match '
                      'weights file shape {}').format( key, v1.shape, v.shape)

    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(checkpoint['state_dict'])
    model.load_state_dict(model_dict, strict=False)

    ckpt_keys = set(checkpoint['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    '''
    for k in missing_keys:
      print('missing keys from checkpoint {}: {}'.format(model_path, k))
    '''
    #print("=> loaded model from checkpoint '{}'".format(model_path))
