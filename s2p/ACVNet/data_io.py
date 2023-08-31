import numpy as np
import re
import torchvision.transforms as transforms
import rasterio


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def quantization(band, percent=1.0):
    a = np.percentile(band, percent)
    b = np.percentile(band, 100 - percent)
    return np.round(255 * (np.clip(band, a, b) - a) / (b - a)).astype(np.uint8)

def load_image(filename):
    img = rasterio.open(filename)
    if img.count == 3:
        return np.dstack([quantization(img.read(1)),
                          quantization(img.read(2)),
                          quantization(img.read(3))])
    if img.count == 1:
        pan = quantization(img.read(1))
        return np.dstack([pan, pan, pan])


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


if __name__ == "__main__":

    pass