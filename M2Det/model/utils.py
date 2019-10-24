import torch.nn.functional as F


def get_backbone():
    pass


def upsample_add(x, y, fuse_type='interp'):
    _, _, H, W = y.size()
    if fuse_type == 'interp':
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    else:
        raise NotImplementedError
