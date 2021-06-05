from torchvision.datasets import STL10
from utils import STL10_ID
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def calc_distortaion(id_type='strip', id_weight=0.75, strip_len=64):
    ds_id = STL10_ID(root='./data', split='train+unlabeled', download=True,
                     id_type=id_type,
                     id_weight=id_weight,
                     strip_len=strip_len)
    ds = STL10(root='./data', split='train+unlabeled')
    l2_list = []
    for (x, y), (x_id, y) in tqdm(zip(ds, ds_id)):
        x = np.asarray(x).astype(np.float32) / 255.0
        x_id = np.asarray(x_id).astype(np.float32) / 255.0
        l2 = np.sqrt(((x - x_id) ** 2).sum())
        l2_list.append(l2)
    l2 = np.mean(l2_list)
    print('L2:', l2)
    return l2


if __name__ == '__main__':
    # calc_distortaion(id_type='2d', id_weight=1.0, strip_len=96)
    # calc_distortaion(id_type='2d', id_weight=0.75, strip_len=96)
    # calc_distortaion(id_type='2d', id_weight=0.5, strip_len=96)
    # calc_distortaion(id_type='2d', id_weight=0.25, strip_len=96)

    calc_distortaion(id_type='strip-hv', id_weight=0.75, strip_len=4)
    for w in [0.25]:
        for l in [64,32,8,4]:
            print(w,l)
            calc_distortaion(id_type='strip-hv', id_weight=w, strip_len=l)
