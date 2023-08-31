import numpy as np
import torch
from scipy import ndimage


def tile_image(img, tile_width, tile_height):

    r""" tile_image
    Tiles an image (tiles overlap if width not divisible by tile_width or
    height not divisible by tile_height)
    输入torch类型: C x H x W

    Parameters
    ----------
    img : Image, numpy array
    tile_width : int
    tile_height: int

    Returns
    -------
    tiles: List of (tile_height x tile_width) numpy arrays of same type as img.
    输出: B x C x H x W, B表示分为B块
    tile_origins: (Nx2) numpy array with the origins (row, col) of the tiles
    输出: B x 2
    """
    img_channel = img.shape[0]
    img_height = img.shape[1]
    img_width = img.shape[2]
    number_of_vertical_tiles = int(np.ceil(img_height / tile_height))
    number_of_horizontal_tiles = int(np.ceil(img_width / tile_width))

    if number_of_vertical_tiles==1:
        total_vertical_overlap = 0
        tile_height = img_height
    else:
        total_vertical_overlap = number_of_vertical_tiles * tile_height - img_height
        tile_vertical_overlap = total_vertical_overlap // (number_of_vertical_tiles - 1)

    if number_of_horizontal_tiles == 1:
        total_horizontal_overlap = 0
        tile_width = img_width
    else:
        total_horizontal_overlap = number_of_horizontal_tiles * tile_width - img_width
        tile_horizontal_overlap = total_horizontal_overlap // (number_of_horizontal_tiles - 1)

    B = number_of_vertical_tiles * number_of_horizontal_tiles
    tile_origins = np.zeros((B, 2), dtype=int)
    tiles = torch.zeros((B, img_channel, tile_height, tile_width), dtype=img.dtype)

    k=0
    for i in range(number_of_vertical_tiles):
        for j in range(number_of_horizontal_tiles):
            if i==0:
                tile_origin_row = 0
            elif i==number_of_vertical_tiles-1:
                tile_origin_row = img_height - tile_height
            else:
                tile_origin_row = i * (tile_height - tile_vertical_overlap)
            if j==0:
                tile_origin_col = 0
            elif j==number_of_horizontal_tiles-1:
                tile_origin_col = img_width - tile_width
            else:
                tile_origin_col = j * (tile_width - tile_horizontal_overlap)

            tile_origins[k,:] = [tile_origin_row, tile_origin_col]
            tiles[k, :, :, :] = img[:, int(tile_origins[k,0]):int(tile_origins[k,0]+tile_height), int(tile_origins[k,1]):int(tile_origins[k,1]+tile_width)]

            k += 1

    return tiles, tile_origins




def untile_image(tiles, tile_origins):
    r""" tile_image
    Rebuild an image from the tiles. If tiles overlap, weight the contribution

    Parameters
    ----------
    tiles: List of (tile_height x tile_width) numpy arrays
    输入torch类型: B x H x W
    tile_origins: (Nx2) numpy array with the origins (row, col) of the tiles
    tile_width : int
    tile_height: int

    Returns
    -------
    img: Rebuilt image
    """

    batch, tile_height, tile_width = tiles.shape

    img_height = tile_origins[-1,0] + tile_height
    img_width =  tile_origins[-1,1] + tile_width

    # when overlapping, the pixels of the inside of the tile are more important than
    # the pixels near the border
    tile_border = np.zeros((tile_height, tile_width))
    tile_border[1:-1,1:-1] = 1
    edt = ndimage.distance_transform_edt(tile_border)
    tile_weight = edt + 0.01
    assert(np.sum(tile_weight==0)==0)

    img = np.zeros((img_height, img_width))
    img_weight = np.zeros((img_height, img_width))

    for k in range(batch):
        img[int(tile_origins[k,0]):int(tile_origins[k,0]+tile_height), int(tile_origins[k,1]):int(tile_origins[k,1]+tile_width)] += tiles[k, :, :] * tile_weight
        img_weight[int(tile_origins[k,0]):int(tile_origins[k,0]+tile_height), int(tile_origins[k,1]):int(tile_origins[k,1]+tile_width)] += tile_weight
    return (img/img_weight).astype(tiles.dtype)